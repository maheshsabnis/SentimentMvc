using Microsoft.Data.SqlClient;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace SentimentMvc.Ml;

// Step 1: Define the schema.
/// <summary>
/// One training/prediction row. Text is the input; Positive is the answer
/// (mapped to the "Label" column that ML.NET binary trainers expect).
/// </summary>
public class SentimentRow
{
    public string Text { get; set; } = "";
    [ColumnName("Label")] public bool Positive { get; set; }
}

/// <summary>
/// What the model returns for one text. PredictedLabel is true for Positive,
/// Probability is the chance of Positive (0 to 1).
/// </summary>
public class SentimentPrediction
{
    [ColumnName("PredictedLabel")] public bool Positive { get; set; }
    public float Probability { get; set; }
    public float Score { get; set; }
}

/// <summary>
/// Baseline vs real-model numbers produced by Step 9 and shown on the page.
/// </summary>
public class ComparisonResult
{
    public double BaselineAccuracy { get; set; }
    public double BaselineAuc { get; set; }
    public double ModelAccuracy { get; set; }
    public double ModelAuc { get; set; }
    public double ModelF1 { get; set; }
    public bool ModelBeatsBaseline { get; set; }
}

/// <summary>
/// Steps 1 to 10 of the sentiment pipeline in one class (step 1 is the schema above).
/// Step 11 (predict) is done by PredictionEnginePool through SentimentService.
/// Create a new instance for every training run; it holds the results of that run.
/// </summary>
public class SentimentTrainer
{
    private readonly MLContext _ml;

    public SentimentTrainer(MLContext ml) => _ml = ml;

    /// <summary>Metrics comparing the baseline with the real model (Step 9).</summary>
    public ComparisonResult Comparison { get; private set; } = null!;

    /// <summary>Number of rows read from SQL Server in this run.</summary>
    public int RowCount { get; private set; }

    /// <summary>Runs steps 2 to 10 in order and writes the model zip to modelPath.</summary>
    public void Run(string connectionString, string modelPath)
    {
        var data = Step02_LoadData(connectionString);
        var split = Step03_TrainTestSplit(data);
        var featurize = Step04_FeaturizeText();

        // Baseline first: the real model must beat it to be worth using
        var baseline = Step05_PriorTrainer(featurize);
        var baselineMetrics = Step06_EvaluateBaseline(baseline, split.TrainSet, split.TestSet);

        var model = Step07_SdcaLogisticRegression(featurize, split.TrainSet);
        var metrics = Step08_EvaluateMetrics(model, split.TestSet);

        Comparison = Step09_CompareWithBaseline(baselineMetrics, metrics);
        Step10_SaveModel(model, split.TrainSet.Schema, modelPath);
    }

    /// <summary>
    /// Cheap "did the table change?" check used by the background worker.
    /// COUNT catches deletes, MAX(Id) catches inserts. NOLOCK means it never blocks
    /// (or is blocked by) writers; an approximate answer is fine for change detection.
    /// It does not read any text, so it adds almost no load on SQL Server.
    /// </summary>
    public static (int Count, int MaxId) GetDataStamp(string connectionString)
    {
        using var con = new SqlConnection(connectionString);
        con.Open();
        using var cmd = new SqlCommand(
            "SELECT COUNT(*), ISNULL(MAX(Id), 0) FROM dbo.SentimentData WITH (NOLOCK)", con);
        using var rd = cmd.ExecuteReader();
        rd.Read();
        return (rd.GetInt32(0), rd.GetInt32(1));
    }

    // Step 2: Load data from SQL Server (mlnet database, SentimentData table).
    private IDataView Step02_LoadData(string connectionString)
    {
        var rows = new List<SentimentRow>();
        using var con = new SqlConnection(connectionString);
        con.Open();
        using var cmd = new SqlCommand("SELECT SentimentText, Sentiment FROM dbo.SentimentData", con);
        using var rd = cmd.ExecuteReader();
        while (rd.Read())
            rows.Add(new SentimentRow { Text = rd.GetString(0), Positive = rd.GetBoolean(1) });

        RowCount = rows.Count;
        return _ml.Data.LoadFromEnumerable(rows);
    }

    // Step 3: 80% for training, 20% for testing. The fixed seed keeps the split repeatable.
    private DataOperationsCatalog.TrainTestData Step03_TrainTestSplit(IDataView data)
        => _ml.Data.TrainTestSplit(data, testFraction: 0.2, seed: 1);

    // Step 4: Turn the text into a numeric "Features" column the trainers can use.
    private IEstimator<ITransformer> Step04_FeaturizeText()
        => _ml.Transforms.Text.FeaturizeText("Features", nameof(SentimentRow.Text));

    // Step 5: Baseline. The Prior trainer just guesses the most common label.
    private IEstimator<ITransformer> Step05_PriorTrainer(IEstimator<ITransformer> featurize)
        => featurize.Append(_ml.BinaryClassification.Trainers.Prior(labelColumnName: "Label"));

    // Step 6: Train the baseline and evaluate it on the test set.
    // Prior is not calibrated, so EvaluateNonCalibrated is used.
    private BinaryClassificationMetrics Step06_EvaluateBaseline(IEstimator<ITransformer> baseline, IDataView train, IDataView test)
    {
        var model = baseline.Fit(train);
        return _ml.BinaryClassification.EvaluateNonCalibrated(model.Transform(test), labelColumnName: "Label");
    }

    // Step 7: The real trainer. SDCA Logistic Regression on the featurized text.
    private ITransformer Step07_SdcaLogisticRegression(IEstimator<ITransformer> featurize, IDataView train)
    {
        var pipeline = featurize.Append(
            _ml.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));
        return pipeline.Fit(train);
    }

    // Step 8: Evaluate the trained model on the test set (data it has never seen).
    private CalibratedBinaryClassificationMetrics Step08_EvaluateMetrics(ITransformer model, IDataView test)
        => _ml.BinaryClassification.Evaluate(model.Transform(test), labelColumnName: "Label");

    // Step 9: Compare the real model with the baseline.
    private ComparisonResult Step09_CompareWithBaseline(BinaryClassificationMetrics baseline, CalibratedBinaryClassificationMetrics model)
        => new()
        {
            BaselineAccuracy = baseline.Accuracy,
            BaselineAuc = baseline.AreaUnderRocCurve,
            ModelAccuracy = model.Accuracy,
            ModelAuc = model.AreaUnderRocCurve,
            ModelF1 = model.F1Score,
            ModelBeatsBaseline = model.AreaUnderRocCurve > baseline.AreaUnderRocCurve
        };

    // Step 10: Save the model. It is written to a temp file first and then moved over the
    // real file, so PredictionEnginePool (which watches the file and reloads it) never sees
    // a half-written zip.
    private void Step10_SaveModel(ITransformer model, DataViewSchema schema, string path)
    {
        var tempPath = path + ".tmp";
        _ml.Model.Save(model, schema, tempPath);
        File.Move(tempPath, path, overwrite: true);
    }
}
