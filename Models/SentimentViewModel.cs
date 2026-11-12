using SentimentMvc.Ml;

namespace SentimentMvc.Models;

/// <summary>
/// Everything the home page needs: the user's text, the result, and the model info.
/// </summary>
public class SentimentViewModel
{
    public string Text { get; set; } = "";

    /// <summary>null = nothing analysed yet, true = Positive, false = Negative.</summary>
    public bool? IsPositive { get; set; }

    /// <summary>Confidence (0 to 1) in the label that is shown.</summary>
    public float Probability { get; set; }

    /// <summary>Baseline vs model metrics from the latest training run.</summary>
    public ComparisonResult? Comparison { get; set; }
    public int RowCount { get; set; }
    public DateTime? LastTrainedAt { get; set; }
}
