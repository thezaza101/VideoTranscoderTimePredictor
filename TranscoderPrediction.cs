using Microsoft.ML.Data;

namespace VideoTranscoderTimePredictor
{
    public class TranscoderPrediction
    {
        [ColumnName("Score")]
        public float PredictedTime;
    }    
}