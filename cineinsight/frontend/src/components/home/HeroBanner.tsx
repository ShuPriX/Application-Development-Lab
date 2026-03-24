import { motion } from "framer-motion";
import SearchBar from "../search/SearchBar";
import FlipWords from "../ui/FlipWords";
import TextGenerateEffect from "../ui/TextGenerateEffect";

export default function HeroBanner() {
  return (
    <div className="relative min-h-[85vh] flex items-center justify-center overflow-hidden">
      {/* Animated background grid */}
      <div className="absolute inset-0 bg-netflix-black" />
      <div
        className="absolute inset-0 opacity-[0.03]"
        style={{
          backgroundImage: `radial-gradient(circle at 1px 1px, white 1px, transparent 0)`,
          backgroundSize: "40px 40px",
        }}
      />
      {/* Radial glow */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_60%_50%_at_50%_40%,rgba(229,9,20,0.12),transparent_70%)]" />

      <div className="relative z-10 text-center px-6 max-w-4xl">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
        >
          {/* Main title */}
          <h1 className="text-5xl md:text-7xl font-extrabold text-white mb-2 tracking-tight leading-tight">
            <FlipWords
              words={["Discover", "Analyze", "Explore", "Decode"]}
              className="text-netflix-red"
              duration={2500}
            />{" "}
            Movies
          </h1>
          <h2 className="text-3xl md:text-5xl font-bold text-white/80 mb-6 tracking-tight">
            Like Never Before
          </h2>
        </motion.div>

        <TextGenerateEffect
          words="AI-powered sentiment analysis of real movie reviews using BERT and BiLSTM deep learning models"
          className="text-gray-400 text-lg md:text-xl mb-12 font-light leading-relaxed"
          delay={400}
        />

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.8 }}
        >
          <SearchBar />
        </motion.div>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.5, duration: 0.8 }}
          className="text-gray-600 text-sm mt-8"
        >
          Powered by BERT + BiLSTM deep learning models
        </motion.p>
      </div>
    </div>
  );
}
