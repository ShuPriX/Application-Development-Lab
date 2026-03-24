import { Card, Chip } from "@heroui/react";
import { motion } from "framer-motion";
import { Brain, Cpu, Database, Globe, Award } from "lucide-react";

const techStack = {
  Frontend: { icon: Globe, items: ["React 19", "TypeScript", "HeroUI", "TailwindCSS", "Recharts", "Framer Motion"] },
  Backend: { icon: Database, items: ["FastAPI", "SQLAlchemy", "httpx", "BeautifulSoup4", "Pydantic"] },
  "Deep Learning": { icon: Brain, items: ["PyTorch", "HuggingFace Transformers", "BERT", "BiLSTM", "GloVe"] },
  "Data Sources": { icon: Globe, items: ["TMDB API", "IMDB Reviews", "Rotten Tomatoes"] },
};

export default function About() {
  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 pt-24 pb-12 min-h-screen">
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <h1 className="text-white text-3xl sm:text-4xl lg:text-5xl font-extrabold mb-4">
          About <span className="text-netflix-red">CineInsight</span>
        </h1>
        <p className="text-gray-400 text-base sm:text-lg mb-10 max-w-2xl leading-relaxed">
          CineInsight is a deep learning-powered movie review analysis system. It scrapes real-time reviews, analyzes sentiment using BERT and BiLSTM models, and provides a comprehensive verdict on whether a movie is worth watching.
        </p>

        <h2 className="flex items-center gap-2 text-white text-xl sm:text-2xl font-bold mb-6">
          <Cpu className="w-6 h-6 text-netflix-red" />
          Model Architecture
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
          <Card className="bg-white/5 border border-white/5 p-5 sm:p-6">
            <div className="flex items-center gap-2 mb-3">
              <Brain className="w-5 h-5 text-netflix-red" />
              <h3 className="text-netflix-red font-bold text-base sm:text-lg">Fine-tuned BERT</h3>
            </div>
            <p className="text-gray-400 text-sm sm:text-base mb-3">Overall Sentiment Classification</p>
            <ul className="text-gray-300 text-sm sm:text-base space-y-1.5">
              <li>Base: bert-base-uncased</li>
              <li>Fine-tuned on IMDB 50K dataset</li>
              <li>3-class: positive / negative / neutral</li>
              <li>Training: AdamW, lr=2e-5, 3 epochs</li>
              <li>Expected accuracy: ~93-95%</li>
            </ul>
          </Card>
          <Card className="bg-white/5 border border-white/5 p-5 sm:p-6">
            <div className="flex items-center gap-2 mb-3">
              <Cpu className="w-5 h-5 text-netflix-red" />
              <h3 className="text-netflix-red font-bold text-base sm:text-lg">BiLSTM with Attention</h3>
            </div>
            <p className="text-gray-400 text-sm sm:text-base mb-3">Aspect-Based Sentiment Analysis</p>
            <ul className="text-gray-300 text-sm sm:text-base space-y-1.5">
              <li>Embedding: 300-dim GloVe</li>
              <li>BiLSTM: 2 layers, 256 hidden units</li>
              <li>5 aspect heads: Acting, Plot, Visuals, Music, Direction</li>
              <li>Training: Adam, lr=1e-3, 15-20 epochs</li>
              <li>With attention mechanism</li>
            </ul>
          </Card>
        </div>

        <h2 className="flex items-center gap-2 text-white text-xl sm:text-2xl font-bold mb-6">
          <Database className="w-6 h-6 text-netflix-red" />
          Tech Stack
        </h2>
        <div className="space-y-6 mb-12">
          {Object.entries(techStack).map(([category, { icon: Icon, items }]) => (
            <div key={category}>
              <h3 className="flex items-center gap-2 text-gray-300 text-sm sm:text-base font-semibold mb-3">
                <Icon className="w-4 h-4 text-gray-500" />
                {category}
              </h3>
              <div className="flex flex-wrap gap-2">
                {items.map((tech) => (
                  <Chip key={tech} variant="bordered" className="text-gray-300 border-gray-600 text-xs sm:text-sm" size="sm">
                    {tech}
                  </Chip>
                ))}
              </div>
            </div>
          ))}
        </div>

        <h2 className="flex items-center gap-2 text-white text-xl sm:text-2xl font-bold mb-6">
          <Award className="w-6 h-6 text-netflix-red" />
          Verdict System
        </h2>
        <Card className="bg-white/5 border border-white/5 p-5 sm:p-6">
          <div className="space-y-3">
            {[
              { verdict: "Strongly Recommended", range: "> 0.8", color: "text-green-400" },
              { verdict: "Worth Watching", range: "> 0.6", color: "text-emerald-400" },
              { verdict: "Mixed Reviews", range: "0.4 - 0.6", color: "text-yellow-400" },
              { verdict: "Below Average", range: "0.2 - 0.4", color: "text-orange-400" },
              { verdict: "Skip It", range: "< 0.2", color: "text-red-400" },
            ].map(({ verdict, range, color }) => (
              <div key={verdict} className="flex flex-col sm:flex-row sm:items-center gap-1 sm:gap-4">
                <span className={`${color} font-semibold text-sm sm:text-base sm:w-56`}>{verdict}</span>
                <span className="text-gray-500 text-sm">Weighted positive score {range}</span>
              </div>
            ))}
          </div>
        </Card>

        <p className="text-gray-600 text-xs sm:text-sm mt-12 text-center">
          Deep Learning Project — Academic Use Only
        </p>
      </motion.div>
    </div>
  );
}
