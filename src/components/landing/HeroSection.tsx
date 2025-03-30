import React from "react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";

const HeroSection = () => {
  return (
    <section className="relative h-screen flex items-center justify-center bg-gradient-to-b from-farmer-50 to-white overflow-hidden">
      <div className="absolute inset-0 bg-[url('https://images.unsplash.com/photo-1500382017468-9049fed747ef?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2532&q=80')] bg-cover bg-center opacity-30"></div>
      
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="container mx-auto px-4 text-center z-10"
      >
        <h1 className="font-extrabold text-5xl md:text-7xl lg:text-8xl mb-6 tracking-tight text-farmer-950">
          <span className="bg-clip-text text-transparent bg-gradient-to-r from-farmer-800 to-farmer-600">
            FARMERS CHOICE
          </span>
        </h1>
        <p className="text-xl md:text-2xl max-w-3xl mx-auto mb-10 text-gray-700">
          AI-powered sustainable farming solutions to optimize resources, increase yields, and improve livelihoods
        </p>
        <div className="flex flex-col md:flex-row gap-4 justify-center">
          <Button 
            size="lg" 
            className="bg-farmer-600 hover:bg-farmer-700 text-white transition-all transform hover:scale-105"
          >
            Get Started
          </Button>
          <Button 
            variant="outline" 
            size="lg"
            className="border-farmer-600 text-farmer-700 hover:bg-farmer-50"
          >
            Learn More
          </Button>
        </div>
      </motion.div>
      
      <motion.div 
        className="absolute bottom-10 left-1/2 transform -translate-x-1/2"
        animate={{ y: [0, 10, 0] }}
        transition={{ repeat: Infinity, duration: 1.5 }}
      >
        <div className="w-8 h-8 rounded-full border-2 border-farmer-600 flex items-center justify-center">
          <div className="w-1 h-4 bg-farmer-600"></div>
        </div>
      </motion.div>
    </section>
  );
};

export default HeroSection;