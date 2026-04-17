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
    // jak czesto dana decyzja wystepuje w stosunku do calego zbioru treningowego
    // _priorProbabilities = 
    // {
    //  "yes": 0.32,
    //  "no": 0.68
    // }
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

    /* {
        "yes": {
            "outlook": {
                "sunny": 0.222,
                "overcast": 0.444,
                "rainy": 0.333
            },
            "temperature": {
                "hot": 0.222,
            },
        "no": {
        }
    }*/
    private void CalculateConditionalProbabilities(List<Observation> dataset)
    {
        var featureNames = dataset.First().Features.Keys; // ["Nogi", "Futro", ... ,]
        var decisionValues = _priorProbabilities.Keys; // ["Tak, "Nie"]
        
        var uniqueValuesCount = featureNames.ToDictionary(
            name => name,
            name => dataset.Select(o => o.Features[name]).Distinct().Count()
        );
        // => { "outlook": 3, ... } 
        
        var allPossibleValues = featureNames.ToDictionary(
            name => name,
            name => dataset.Select(o => o.Features[name]).Distinct().ToList()
        );
        // => { "outlook": ["sunny", "overcast", "rainy"], ... } 
        
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

    }

    public string Predict(Observation toPredict)
    {
        string bestDecision = null;
        double highestScore = -1.0;

        foreach (var decision in _priorProbabilities.Keys)
        {
            double currentScore = _priorProbabilities[decision];

            foreach (var feature in toPredict.Features)
            {
                string featureName = feature.Key;
                string featureValue = feature.Value;
                
                if (_conditionalProbabilities[decision].ContainsKey(featureName) &&
                    _conditionalProbabilities[decision][featureName].ContainsKey(featureValue))
                {
                    currentScore *= _conditionalProbabilities[decision][featureName][featureValue];
                }
            }
            
            if (currentScore > highestScore)
            {
                highestScore = currentScore;
                bestDecision = decision;
            }
        }

        return bestDecision;
    }

    public double SimpleSmoothing(int numerator, int denominator, int classes)
    {
        return (double)(numerator + 1) / (denominator + classes);
    }


}