namespace NaiveBayes
{
    public class Observation
    {
        public Dictionary<string, string> Features { get; set; }
        public string Decision { get; set; }
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


        }
    }
}