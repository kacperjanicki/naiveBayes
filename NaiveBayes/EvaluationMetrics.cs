namespace NaiveBayes;

public class EvaluationMetrics
{
    public double MeasureAccuracy(List<string> realLabels, List<string> predictedLabels)
    {
        if (realLabels.Count == 0) return 0;

        int correct = 0;
        for (int i = 0; i < realLabels.Count; i++)
        {
            if (realLabels[i] == predictedLabels[i])
            {
                correct++;
            }
        }

        return (double)correct / realLabels.Count;
    }
    // Z tych które model zakwalifikował jako TAK, ile faktycznie było TAK
    public double MeasurePrecision(List<string> realLabels, List<string> predictedLabels, string targetClass)
    {
        int truePositive = 0;
        int falsePositive = 0;

        for (int i = 0; i < realLabels.Count; i++)
        {
            if (predictedLabels[i] == targetClass)
            {
                if (realLabels[i] == targetClass) truePositive++;
                else falsePositive++;
            }
        }

        return (truePositive + falsePositive) == 0 ? 0 : (double)truePositive / (truePositive + falsePositive);
    }

    // Z tych, które faktycznie były TAK ile udało się wyłapać
    // FN - false negative, było TAK a model powiedział NIE
    public double MeasureRecall(List<string> realLabels, List<string> predictedLabels, string targetClass)
    {
        int truePositive = 0;
        int falseNegative = 0;

        for (int i = 0; i < realLabels.Count; i++)
        {
            if (realLabels[i] == targetClass)
            {
                if (predictedLabels[i] == targetClass) truePositive++;
                else falseNegative++;
            }
        }

        return (truePositive + falseNegative) == 0 ? 0 : (double)truePositive / (truePositive + falseNegative);
    }
    // Miara, które sprawdza czy model był dobry w dwóch metrykach jednoczesnie
    public double MeasureFMeasure(double precision, double recall)
    {
        if (precision + recall == 0) return 0;
        return 2 * (precision * recall) / (precision + recall);
    }
}