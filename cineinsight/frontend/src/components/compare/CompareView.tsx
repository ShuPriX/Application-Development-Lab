import {
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Legend,
  Tooltip,
} from "recharts";
import VerdictBadge from "../analysis/VerdictBadge";
import SentimentGauge from "../analysis/SentimentGauge";
import type { AnalysisResult } from "../../types/analysis";

interface CompareViewProps {
  analysisA: AnalysisResult;
  analysisB: AnalysisResult;
}

export default function CompareView({
  analysisA,
  analysisB,
}: CompareViewProps) {
  const radarData = [
    {
      aspect: "Acting",
      [analysisA.movie.title]: analysisA.aspect_scores.acting * 100,
      [analysisB.movie.title]: analysisB.aspect_scores.acting * 100,
    },
    {
      aspect: "Plot",
      [analysisA.movie.title]: analysisA.aspect_scores.plot * 100,
      [analysisB.movie.title]: analysisB.aspect_scores.plot * 100,
    },
    {
      aspect: "Visuals",
      [analysisA.movie.title]: analysisA.aspect_scores.visuals * 100,
      [analysisB.movie.title]: analysisB.aspect_scores.visuals * 100,
    },
    {
      aspect: "Music",
      [analysisA.movie.title]: analysisA.aspect_scores.music * 100,
      [analysisB.movie.title]: analysisB.aspect_scores.music * 100,
    },
    {
      aspect: "Direction",
      [analysisA.movie.title]: analysisA.aspect_scores.direction * 100,
      [analysisB.movie.title]: analysisB.aspect_scores.direction * 100,
    },
  ];

  const totalA =
    analysisA.review_sentiments?.length
      ? {
          positive: analysisA.review_sentiments.filter(
            (r) => r.sentiment_label === "positive",
          ).length,
          neutral: analysisA.review_sentiments.filter(
            (r) => r.sentiment_label === "neutral",
          ).length,
          negative: analysisA.review_sentiments.filter(
            (r) => r.sentiment_label === "negative",
          ).length,
        }
      : analysisA.source_comparison.imdb ?? { positive: 0, neutral: 0, negative: 0 };

  const totalB =
    analysisB.review_sentiments?.length
      ? {
          positive: analysisB.review_sentiments.filter(
            (r) => r.sentiment_label === "positive",
          ).length,
          neutral: analysisB.review_sentiments.filter(
            (r) => r.sentiment_label === "neutral",
          ).length,
          negative: analysisB.review_sentiments.filter(
            (r) => r.sentiment_label === "negative",
          ).length,
        }
      : analysisB.source_comparison.imdb ?? { positive: 0, neutral: 0, negative: 0 };

  return (
    <div className="space-y-8">
      {/* Verdicts side-by-side */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <p className="text-gray-400 text-sm text-center mb-3">
            {analysisA.movie.title} ({analysisA.movie.year})
          </p>
          <VerdictBadge
            verdict={analysisA.verdict}
            confidence={analysisA.confidence}
            sentiment={analysisA.overall_sentiment}
          />
        </div>
        <div>
          <p className="text-gray-400 text-sm text-center mb-3">
            {analysisB.movie.title} ({analysisB.movie.year})
          </p>
          <VerdictBadge
            verdict={analysisB.verdict}
            confidence={analysisB.confidence}
            sentiment={analysisB.overall_sentiment}
          />
        </div>
      </div>

      {/* Combined radar chart */}
      <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-6 border border-white/5">
        <h3 className="text-white font-semibold text-sm mb-4">
          Aspect Comparison
        </h3>
        <ResponsiveContainer width="100%" height={350}>
          <RadarChart data={radarData}>
            <PolarGrid stroke="#333" />
            <PolarAngleAxis
              dataKey="aspect"
              tick={{ fill: "#9ca3af", fontSize: 12 }}
            />
            <PolarRadiusAxis
              angle={90}
              domain={[0, 100]}
              tick={{ fill: "#6b7280", fontSize: 10 }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#1a1a1a",
                border: "1px solid #333",
                borderRadius: "8px",
                color: "#fff",
              }}
            />
            <Legend wrapperStyle={{ color: "#9ca3af", fontSize: 12 }} />
            <Radar
              name={analysisA.movie.title}
              dataKey={analysisA.movie.title}
              stroke="#E50914"
              fill="#E50914"
              fillOpacity={0.15}
              strokeWidth={2}
            />
            <Radar
              name={analysisB.movie.title}
              dataKey={analysisB.movie.title}
              stroke="#3B82F6"
              fill="#3B82F6"
              fillOpacity={0.15}
              strokeWidth={2}
            />
          </RadarChart>
        </ResponsiveContainer>
      </div>

      {/* Sentiment comparison */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <SentimentGauge distribution={totalA} />
        <SentimentGauge distribution={totalB} />
      </div>
    </div>
  );
}
