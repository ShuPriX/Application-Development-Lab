import { BrowserRouter, Routes, Route } from "react-router-dom";
import { HeroUIProvider } from "@heroui/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AnimatePresence } from "framer-motion";
import Navbar from "./components/common/Navbar";
import Footer from "./components/common/Footer";
import Home from "./pages/Home";
import Analysis from "./pages/Analysis";
import Compare from "./pages/Compare";
import About from "./pages/About";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <HeroUIProvider>
        <BrowserRouter>
          <div className="dark min-h-screen bg-background text-foreground">
            <Navbar />
            <main>
              <AnimatePresence mode="wait">
                <Routes>
                  <Route path="/" element={<Home />} />
                  <Route path="/analysis/:tmdbId" element={<Analysis />} />
                  <Route path="/compare" element={<Compare />} />
                  <Route path="/about" element={<About />} />
                </Routes>
              </AnimatePresence>
            </main>
            <Footer />
          </div>
        </BrowserRouter>
      </HeroUIProvider>
    </QueryClientProvider>
  );
}

export default App;
