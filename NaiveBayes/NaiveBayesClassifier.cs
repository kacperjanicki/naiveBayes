namespace NaiveBayes;

public class NaiveBayesClassifier
{
    private bool _applySmoothingAll;
    private Dictionary<string, double> _priorProbabilities;
    private Dictionary<string, Dictionary<string, Dictionary<string, double>>> _conditionalProbabilities;
    
    public NaiveBayesClassifier(bool applySmoothingAll, List<Observation> trainDataset)
    {
        _applySmoothingAll = applySmoothingAll;
        _conditionalProbabilities = new Dictionary<string, Dictionary<string, Dictionary<string, double>>>();
        _priorProbabilities = new Dictionary<string, double>();
        CalculatePriorProbabilities(trainDataset);
        CalculateConditionalProbabilities(trainDataset);

    }
    
    private void CalculatePriorProbabilities(List<Observation> dataset)
    {
        int allOccurences = dataset.Count;
        var decisions = dataset.Select(o => o.Decision).Distinct();

        foreach (var d in decisions)
        {
            _priorProbabilities[d] = 0;
        }
        
        foreach (var observation in dataset)
        {
            _priorProbabilities[observation.Decision] += 1;
        }

        foreach (var pair in _priorProbabilities)
        {
            _priorProbabilities[pair.Key] = pair.Value / allOccurences;
        }
    }

    private void CalculateConditionalProbabilities(List<Observation> dataset)
    {
        var featureNames = dataset.First().Features.Keys; // ["Nogi", "Jaja", ... ,]
        var decisionValues = _priorProbabilities.Keys; // ["Tak, "Nie"]
        
        var uniqueValuesCount = featureNames.ToDictionary(
            name => name,
            name => dataset.Select(o => o.Features[name]).Distinct().Count()
        );
        
        var allPossibleValues = featureNames.ToDictionary(
            name => name,
            name => dataset.Select(o => o.Features[name]).Distinct().ToList()
        );
        
        foreach (var VARIABLE in allPossibleValues)
        {
            Console.WriteLine(string.Join(VARIABLE.Key, VARIABLE.Value));
        }

        foreach (var decision in decisionValues)
        {
            _conditionalProbabilities[decision] = new Dictionary<string, Dictionary<string, double>>();
            var observationsInClass = dataset.Where(o => o.Decision == decision).ToList();
            int denominator = observationsInClass.Count();

            foreach (var feature in featureNames)
            {
                int classes = uniqueValuesCount[feature];
                _conditionalProbabilities[decision][feature] = new Dictionary<string, double>();

                foreach (var value in allPossibleValues[feature])
                {
                    int numerator = observationsInClass.Count(o => o.Features[feature] == value);
                    double probability;

                    if (_applySmoothingAll)
                    {
                        probability = SimpleSmoothing(numerator, denominator, classes);
                    }
                    else
                    {
                        if (numerator == 0)
                        {
                            probability = SimpleSmoothing(numerator, denominator, classes);
                        }
                        else
                        {
                            probability = (double)numerator / denominator;
                        }
                    }

                    _conditionalProbabilities[decision][feature][value] = probability;
                }
            }

        }

        Console.WriteLine(decisionValues);
    }

    public double SimpleSmoothing(int numerator, int denominator, int classes)
    {
        return (double)(numerator + 1) / (denominator + classes);
    }


}