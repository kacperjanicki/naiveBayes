namespace NaiveBayes
{
    public class Observation
    {
        public Dictionary<string, string> Features { get; set; }
        public string Decision { get; set; }
        
        public override string ToString()
        {
            var featuresList = Features.Select(f => $"{f.Key}={f.Value}");
            string featuresString = string.Join(", ", featuresList);
    
            return $"[Obserwacja] Cechy: ({featuresString}), Decyzja: {Decision}";
        }
    }
    
    class Program
    {
        static void Main(string[] args)
        {
            List<Observation> allData = new List<Observation>
{
                new Observation { Decision = "no", Features = new Dictionary<string, string> { { "outlook", "sunny" }, { "temperature", "hot" }, { "humidity", "high" }, { "windy", "false" } } },
                new Observation { Decision = "no", Features = new Dictionary<string, string> { { "outlook", "sunny" }, { "temperature", "hot" }, { "humidity", "high" }, { "windy", "true" } } },
                new Observation { Decision = "yes", Features = new Dictionary<string, string> { { "outlook", "overcast" }, { "temperature", "hot" }, { "humidity", "high" }, { "windy", "false" } } },
                new Observation { Decision = "yes", Features = new Dictionary<string, string> { { "outlook", "rainy" }, { "temperature", "mild" }, { "humidity", "high" }, { "windy", "false" } } },
                new Observation { Decision = "yes", Features = new Dictionary<string, string> { { "outlook", "rainy" }, { "temperature", "cool" }, { "humidity", "normal" }, { "windy", "false" } } },
                new Observation { Decision = "no", Features = new Dictionary<string, string> { { "outlook", "rainy" }, { "temperature", "cool" }, { "humidity", "normal" }, { "windy", "true" } } },
                new Observation { Decision = "yes", Features = new Dictionary<string, string> { { "outlook", "overcast" }, { "temperature", "cool" }, { "humidity", "normal" }, { "windy", "true" } } },
                new Observation { Decision = "no", Features = new Dictionary<string, string> { { "outlook", "sunny" }, { "temperature", "mild" }, { "humidity", "high" }, { "windy", "false" } } },
                new Observation { Decision = "yes", Features = new Dictionary<string, string> { { "outlook", "sunny" }, { "temperature", "cool" }, { "humidity", "normal" }, { "windy", "false" } } },
                new Observation { Decision = "yes", Features = new Dictionary<string, string> { { "outlook", "rainy" }, { "temperature", "mild" }, { "humidity", "normal" }, { "windy", "false" } } },
                new Observation { Decision = "yes", Features = new Dictionary<string, string> { { "outlook", "sunny" }, { "temperature", "mild" }, { "humidity", "normal" }, { "windy", "true" } } },
                new Observation { Decision = "yes", Features = new Dictionary<string, string> { { "outlook", "overcast" }, { "temperature", "mild" }, { "humidity", "high" }, { "windy", "true" } } },
                new Observation { Decision = "yes", Features = new Dictionary<string, string> { { "outlook", "overcast" }, { "temperature", "hot" }, { "humidity", "normal" }, { "windy", "false" } } },
                new Observation { Decision = "no", Features = new Dictionary<string, string> { { "outlook", "rainy" }, { "temperature", "mild" }, { "humidity", "high" }, { "windy", "true" } } }
            };
            
            int testSize = 2;
            int trainSize = allData.Count - testSize;
            List<Observation> trainDataset = allData.GetRange(0, trainSize);
            List<Observation> testDataset = allData.GetRange(trainSize, testSize);

            NaiveBayesClassifier classifier = new NaiveBayesClassifier(true,allData);
            
            foreach (var testCase in trainDataset)
            {
                string prediction = classifier.Predict(testCase);
                Console.WriteLine($"{testCase}, Przewidziano: {prediction}");
            }
            
            List<string> realLabels = testDataset.Select(o => o.Decision).ToList();
            List<string> predictedLabels = new List<string>();

            foreach (var testCase in trainDataset)
            {
                predictedLabels.Add(classifier.Predict(testCase));
            }

            EvaluationMetrics metrics = new EvaluationMetrics();
            var allClassNames = realLabels.Concat(predictedLabels).Distinct();
            Console.WriteLine("--- Evaluation Metrics ---");
            Console.WriteLine($"Ogolna dokladnosc (Accuracy): {metrics.MeasureAccuracy(realLabels, predictedLabels):P2}");

            foreach (var className in allClassNames)
            {
                double precision = metrics.MeasurePrecision(realLabels, predictedLabels, className);
                double recall = metrics.MeasureRecall(realLabels, predictedLabels, className);
                double fMeasure = metrics.MeasureFMeasure(precision, recall);

                Console.WriteLine($"--- Klasa: {className} ---");
                Console.WriteLine($"Precyzja: {precision:P2}");
                Console.WriteLine($"Pelnosc (Recall): {recall:P2}");
                Console.WriteLine($"F-miara: {fMeasure:P2}");
            }


        }
    }
}