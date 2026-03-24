import { Brain, Film } from "lucide-react";

export default function Footer() {
  return (
    <footer className="border-t border-white/5 bg-netflix-black py-10 mt-auto">
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <Film className="w-5 h-5 text-netflix-red" />
            <span className="text-netflix-red font-bold text-lg">CINE</span>
            <span className="text-white font-light text-lg">INSIGHT</span>
          </div>
          <div className="flex items-center gap-6 text-gray-500 text-sm">
            <span className="flex items-center gap-1.5">
              <Brain className="w-4 h-4" />
              BERT + BiLSTM
            </span>
            <span>Deep Learning Project</span>
          </div>
        </div>
        <p className="text-gray-600 text-xs text-center mt-6">
          Academic Use Only — Movie Review Sentiment Analysis
        </p>
      </div>
    </footer>
  );
}
