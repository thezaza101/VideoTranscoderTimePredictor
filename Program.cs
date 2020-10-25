using System;

namespace VideoTranscoderTimePredictor
{
    class Program
    {
        static void Main(string[] args)
        {
            TranscoderModel model = new TranscoderModel();
            model.LoadData();
            model.BuildPipeline();
            model.BuildAndTrainModel();
            model.EvaluateModel();
            model.SaveModelToFile(TranscoderModel._modelPath);
        }
    }
}
