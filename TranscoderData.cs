using Microsoft.ML.Data;

namespace VideoTranscoderTimePredictor
{
    public class TranscoderData
    {
        [LoadColumn(0),ColumnName("inbitrate")]
        public int inbitrate;

        [LoadColumn(1),ColumnName("incodec")]
        public string incodec;

        [LoadColumn(2),ColumnName("induration")]
        public float induration;

        [LoadColumn(3),ColumnName("inframerate")]
        public float inframerate;

        [LoadColumn(4),ColumnName("inframes")]
        public int inframes;

        [LoadColumn(5),ColumnName("inheight")]
        public int inheight;

        [LoadColumn(6),ColumnName("inwidth")]
        public int inwidth;

        [LoadColumn(7),ColumnName("insize")]
        public int insize;

        [LoadColumn(8),ColumnName("outbitrate")]
        public int outbitrate;

        [LoadColumn(9),ColumnName("outcodec")]
        public string outcodec;

        [LoadColumn(10),ColumnName("outframerate")]
        public float outframerate;

        [LoadColumn(11),ColumnName("outheight")]
        public int outheight;

        [LoadColumn(12),ColumnName("outwidth")]
        public int outwidth;

        [LoadColumn(13),ColumnName("utime")]
        public float utime;
    }
}