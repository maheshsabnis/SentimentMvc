namespace SentimentMvc.Ml;

/// <summary>
/// Background service that handles rows being inserted while the app runs.
/// Every few minutes it runs one tiny query (COUNT and MAX(Id)); only when the
/// table changed does it retrain. Retraining runs on a thread-pool thread, never on a
/// request thread, and predictions keep using the old model until the new one is ready.
/// </summary>
public class RetrainWorker : BackgroundService
{
    private readonly SentimentService _service;
    private readonly IConfiguration _config;
    private readonly ILogger<RetrainWorker> _log;

    public RetrainWorker(SentimentService service, IConfiguration config, ILogger<RetrainWorker> log)
    {
        _service = service;
        _config = config;
        _log = log;
    }

    protected override async Task ExecuteAsync(CancellationToken ct)
    {
        // Interval comes from appsettings.json (Retrain:IntervalMinutes), default 5
        var minutes = _config.GetValue("Retrain:IntervalMinutes", 5);
        using var timer = new PeriodicTimer(TimeSpan.FromMinutes(minutes));

        // The first tick happens after one interval, because Program.cs already trained at startup
        while (await timer.WaitForNextTickAsync(ct))
        {
            try
            {
                if (await Task.Run(_service.TrainIfChanged, ct))
                    _log.LogInformation("SentimentData changed. Model retrained.");
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                // SQL down, training error, etc. The old model keeps serving; we try again next tick.
                _log.LogError(ex, "Retraining failed. Keeping the previous model.");
            }
        }
    }
}
