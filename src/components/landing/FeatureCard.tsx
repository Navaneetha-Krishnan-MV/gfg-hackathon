import React from "react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { LucideIcon } from "lucide-react";

interface FeatureCardProps {
  title: string;
  description: string;
  icon: React.ReactNode;
  color: string;
  index: number;
}

const FeatureCard: React.FC<FeatureCardProps> = ({ title, description, icon, color, index }) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: index * 0.1 }}
      viewport={{ once: true }}
      whileHover={{ 
        y: -15, 
        boxShadow: "0 20px 25px -5px rgba(0, 0, 0, 0.2), 0 10px 10px -5px rgba(0, 0, 0, 0.1)",
        transition: { type: "spring", stiffness: 300, damping: 15 }
      }}
      className={`min-w-[350px] h-[450px] flex-shrink-0 rounded-2xl bg-gradient-to-br ${color} p-8 flex flex-col justify-between border border-gray-800 hover:border-farmer-400 transition-all duration-300`}
    >
      <div>
        <div className="mb-6 transform transition-transform duration-300 hover:scale-110">
          {icon}
        </div>
        <h3 className="text-3xl font-bold mb-4">{title}</h3>
        <p className="text-gray-300 text-lg">{description}</p>
      </div>
      <Button 
        variant="outline" 
        className="w-full mt-6 border-farmer-600 text-farmer-300 hover:bg-farmer-800 hover:text-white hover:border-farmer-300 transition-all duration-300 transform hover:scale-105"
      >
        Learn More
      </Button>
    </motion.div>
  );
};

export default FeatureCard;