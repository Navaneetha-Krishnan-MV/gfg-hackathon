import React from "react";
import { motion } from "framer-motion";
import { Droplets, Leaf, BarChart } from "lucide-react";

const MissionSection = () => {
  return (
    <section className="py-20 bg-white">
      <div className="container mx-auto px-4">
        <div className="flex flex-col md:flex-row gap-10 items-center">
          <motion.div 
            className="md:w-1/2"
            initial={{ opacity: 0, x: -50 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            <h2 className="text-4xl font-bold mb-6 text-gray-900">Our Mission</h2>
            <p className="text-lg text-gray-700 mb-6">
              At FarmTech, we're dedicated to transforming agriculture through data-driven insights and sustainable practices. 
              Our AI-powered platform helps farmers reduce water usage, minimize pesticide application, prevent soil degradation, 
              and lower their carbon footprint while improving yields and profitability.
            </p>
            <div className="flex flex-wrap gap-4">
              <div className="flex items-center gap-2 bg-farmer-50 px-4 py-2 rounded-full">
                <Droplets size={18} className="text-farmer-700" />
                <span className="text-sm font-medium">Water Conservation</span>
              </div>
              <div className="flex items-center gap-2 bg-farmer-50 px-4 py-2 rounded-full">
                <Leaf size={18} className="text-farmer-700" />
                <span className="text-sm font-medium">Sustainable Methods</span>
              </div>
              <div className="flex items-center gap-2 bg-farmer-50 px-4 py-2 rounded-full">
                <BarChart size={18} className="text-farmer-700" />
                <span className="text-sm font-medium">Profit Optimization</span>
              </div>
            </div>
          </motion.div>
          
          <motion.div 
            className="md:w-1/2 bg-black rounded-2xl overflow-hidden shadow-xl"
            initial={{ opacity: 0, x: 50 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            <div className="aspect-video relative">
              <img 
                src="https://images.unsplash.com/photo-1530836369250-ef72a3f5cda8?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2940&q=80" 
                alt="Sustainable Farming" 
                className="w-full h-full object-cover"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/70 to-transparent flex items-end p-6">
                <p className="text-white font-medium">Sustainable farming practices in action</p>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
};

export default MissionSection;