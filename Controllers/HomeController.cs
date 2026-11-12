using System.Diagnostics;
using Microsoft.AspNetCore.Mvc;
using SentimentMvc.Ml;
using SentimentMvc.Models;

namespace SentimentMvc.Controllers;

/// <summary>
/// The only page: the user types a statement and sees Positive or Negative.
/// </summary>
public class HomeController : Controller
{
    private readonly SentimentService _service;
    public HomeController(SentimentService service) => _service = service;

    /// <summary>Shows the empty form plus the current model metrics.</summary>
    [HttpGet]
    public IActionResult Index() => View(BuildModel(""));

    /// <summary>Analyses the text and shows the result (Step 11).</summary>
    [HttpPost]
    public IActionResult Index(string text)
    {
        var vm = BuildModel(text ?? "");
        if (!string.IsNullOrWhiteSpace(text))
        {
            // Limit length so one huge input cannot make a request expensive
            var p = _service.Predict(text.Length > 2000 ? text[..2000] : text);
            vm.IsPositive = p.Positive;
            // Probability is the chance of Positive; show confidence for the label we display
            vm.Probability = p.Positive ? p.Probability : 1 - p.Probability;
        }
        return View(vm);
    }

    /// <summary>
    /// Called by the "Refresh model now" button (page JavaScript, no page reload).
    /// auto = false: the user clicked, so always retrain (also picks up edited rows).
    /// auto = true: the 2-minute timer fired, so only retrain if rows were added or removed.
    /// That check is one light query, so the timer adds almost no load when nothing changed.
    /// </summary>
    [HttpPost]
    public IActionResult Retrain(bool auto = false)
    {
        if (auto)
            _service.TrainIfChanged();
        else
            _service.Train();

        return Ok();
    }

    // Fills the model info shown under the form
    private SentimentViewModel BuildModel(string text) => new()
    {
        Text = text,
        Comparison = _service.Comparison,
        RowCount = _service.RowCount,
        LastTrainedAt = _service.LastTrainedAt
    };

    [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
    public IActionResult Error()
    {
        return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
    }
}
