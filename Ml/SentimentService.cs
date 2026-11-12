using Microsoft.Extensions.ML;
using Microsoft.ML;

namespace SentimentMvc.Ml;

/// <summary>
/// Singleton that owns training (steps 1 to 10 via SentimentTrainer) and prediction (step 11).
/// Prediction goes through PredictionEnginePool, so many requests can predict in parallel
/// with no lock. Training writes a new model zip; the pool notices the file change and
/// reloads it by itself, so users never wait for a retrain.
/// </summary>
public class SentimentService
{
    // Name the model is registered under in Program.cs
    public const string ModelName = "sentiment";

    private readonly PredictionEnginePool<SentimentRow, SentimentPrediction> _pool;
    private readonly string _connectionString;
    private readonly string _modelPath;

    // Only one training run at a time (the timer and the Retrain button could overlap).
    // Predictions never take this lock.
    private readonly object _trainLock = new();

    // The table state the current model was trained on. Compared by the worker.
    private (int Count, int MaxId) _stamp;

    // Written once per training run, read by the page. Volatile so readers see the new object.
    private volatile ComparisonResult? _comparison;
    private int _rowCount;
    private long _lastTrainedTicks;

    public ComparisonResult? Comparison => _comparison;
    public int RowCount => Volatile.Read(ref _rowCount);
    public DateTime? LastTrainedAt
    {
        get
        {
            var t = Interlocked.Read(ref _lastTrainedTicks);
            return t == 0 ? null : new DateTime(t);
        }
    }

    public SentimentService(PredictionEnginePool<SentimentRow, SentimentPrediction> pool,
        IConfiguration config, IWebHostEnvironment env)
    {
        _pool = pool;
        _connectionString = config.GetConnectionString("DefaultConnection")!;
        _modelPath = ModelPath(env);
    }

    /// <summary>Full path of the model zip. Shared with Program.cs so both use the same file.</summary>
    public static string ModelPath(IWebHostEnvironment env)
        => Path.Combine(env.ContentRootPath, "sentiment-model.zip");

    /// <summary>
    /// Runs steps 1 to 10. Called at startup, by the Retrain button and by the worker.
    /// </summary>
    public void Train()
    {
        lock (_trainLock)
        {
            // Take the stamp BEFORE loading data. Rows inserted while we train will make
            // the stamp differ on the next check, so they are picked up next time.
            var stamp = SentimentTrainer.GetDataStamp(_connectionString);

            // Own MLContext per run so nothing is shared with the prediction side
            var trainer = new SentimentTrainer(new MLContext(seed: 1));
            trainer.Run(_connectionString, _modelPath);

            // The model file is already replaced (and the pool will reload it).
            // Now publish the numbers shown on the page.
            _stamp = stamp;
            _comparison = trainer.Comparison;
            Volatile.Write(ref _rowCount, trainer.RowCount);
            Interlocked.Exchange(ref _lastTrainedTicks, DateTime.Now.Ticks);
        }
    }

    /// <summary>
    /// Retrains only if rows were added or removed. The check is one light query, so it is
    /// safe to call often. Returns true if it retrained.
    /// </summary>
    public bool TrainIfChanged()
    {
        var current = SentimentTrainer.GetDataStamp(_connectionString);
        if (current == _stamp) return false;

        Train();
        return true;
    }

    /// <summary>Step 11: Predict a new review. The pool lends an engine for this one call.</summary>
    public SentimentPrediction Predict(string text)
        => _pool.Predict(ModelName, new SentimentRow { Text = text });
}
