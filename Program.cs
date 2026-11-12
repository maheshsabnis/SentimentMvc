using Microsoft.Extensions.ML;
using SentimentMvc.Ml;
var builder = WebApplication.CreateBuilder(args);
// MVC controllers and views
builder.Services.AddControllersWithViews();
// Step 11 support: a pool of prediction engines, so many requests can predict in parallel.
// watchForChanges: true reloads the model automatically when the zip file is replaced
// by a retrain, with no restart and no lock.
var modelPath = Path.Combine(builder.Environment.ContentRootPath, "sentiment-model.zip");
builder.Services.AddPredictionEnginePool<SentimentRow, SentimentPrediction>()
    .FromFile(modelName: SentimentService.ModelName, filePath: modelPath, watchForChanges: true);
// Trains and publishes metrics
builder.Services.AddSingleton<SentimentService>();
// Checks for new rows every few minutes and retrains only if the table changed
builder.Services.AddHostedService<RetrainWorker>();
var app = builder.Build();
// Train once before the first request so the model zip exists (steps 1 to 10).
// This must run before the pool is first used, because the pool loads the zip.
app.Services.GetRequiredService<SentimentService>().Train();
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Home/Error");
}
app.UseRouting();
app.UseAuthorization();
app.MapStaticAssets();
app.MapControllerRoute(
    name: "default",
    pattern: "{controller=Home}/{action=Index}/{id?}")
    .WithStaticAssets();
app.Run();
