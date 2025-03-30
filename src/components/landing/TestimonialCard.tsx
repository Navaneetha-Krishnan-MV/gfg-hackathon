import React from "react";
import { motion } from "framer-motion";

interface TestimonialCardProps {
  quote: string;
  name: string;
  farm: string;
  image: string;
  index: number;
}

const TestimonialCard: React.FC<TestimonialCardProps> = ({ quote, name, farm, image, index }) => {
  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: index * 0.1 }}
      viewport={{ once: true }}
      className="bg-white rounded-xl shadow-lg p-6 hover:shadow-xl transition-shadow"
    >
      <div className="mb-4 text-farmer-600">
        <svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M9.33333 17.3333C11.5425 17.3333 13.3333 15.5425 13.3333 13.3333C13.3333 11.1242 11.5425 9.33333 9.33333 9.33333C7.12422 9.33333 5.33333 11.1242 5.33333 13.3333C5.33333 15.5425 7.12422 17.3333 9.33333 17.3333Z" fill="currentColor"/>
          <path d="M22.6667 17.3333C24.8758 17.3333 26.6667 15.5425 26.6667 13.3333C26.6667 11.1242 24.8758 9.33333 22.6667 9.33333C20.4575 9.33333 18.6667 11.1242 18.6667 13.3333C18.6667 15.5425 20.4575 17.3333 22.6667 17.3333Z" fill="currentColor"/>
          <path d="M9.33333 22.6667C6.17333 22.6667 0 24.25 0 27.4C0 28.84 1.16 29.3333 2.66667 29.3333H16C16 26.86 18.01 24.7933 20.44 24.3333C17.1067 23.0267 12.74 22.6667 9.33333 22.6667Z" fill="currentColor"/>
          <path d="M22.6667 22.6667C22.2267 22.6667 21.7533 22.6933 21.2667 22.73C23.8933 23.98 25.3333 26.1267 25.3333 28.6667C25.3333 30.5067 24.0533 32 22.6667 32C21.28 32 20 30.5067 20 28.6667C20 26.8267 21.28 25.3333 22.6667 25.3333C24.0533 25.3333 25.3333 26.8267 25.3333 28.6667H29.3333C29.3333 25.52 26.3067 22.6667 22.6667 22.6667Z" fill="currentColor"/>
        </svg>
      </div>
      <p className="text-gray-700 mb-6 italic">"{quote}"</p>
      <div className="flex items-center">
        <div className="w-12 h-12 rounded-full overflow-hidden mr-4">
          <img src={image} alt={name} className="w-full h-full object-cover" />
        </div>
        <div>
          <p className="font-semibold text-gray-900">{name}</p>
          <p className="text-sm text-gray-500">{farm}</p>
        </div>
      </div>
    </motion.div>
  );
};

export default TestimonialCard;