import React, { useRef, useEffect } from "react";
import { Sprout, BarChart, Cloud, Droplets, Leaf, Tractor } from "lucide-react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import FeatureCard from "./FeatureCard";

const FeaturesSection = () => {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleWheel = (e: WheelEvent) => {
      if (scrollRef.current && e.target instanceof Node && scrollRef.current.contains(e.target)) {
        // Get element dimensions and position
        const rect = scrollRef.current.getBoundingClientRect();
        const bottomEdge = rect.bottom;
        const isNearBottomEdge = Math.abs(e.clientY - bottomEdge) < 20; // Within 20px of bottom edge
        
        // Move cursor position if it's near the bottom border
        if (isNearBottomEdge) {
          // Calculate a position slightly above center
          const newY = rect.top + (rect.height * 0.40); // 40% from the top (slightly above center)
          
          try {
            // Try to move the cursor if browser supports it
            window.scrollTo({
              top: window.scrollY + (newY - e.clientY),
              behavior: 'smooth'
            });
          } catch (err) {
            // If browser doesn't support smooth scrolling or other issues
            console.log("Could not adjust scroll position:", err);
          }
        }
        
        // Calculate if we're at the ends of the horizontal scroll
        const { scrollLeft, scrollWidth, clientWidth } = scrollRef.current;
        const isAtRightEnd = Math.abs(scrollWidth - scrollLeft - clientWidth) < 1;
        const isAtLeftEnd = scrollLeft <= 0;
        
        // Determine if we should allow default scrolling behavior
        const scrollingLeft = e.deltaY < 0;
        const scrollingRight = e.deltaY > 0;
        
        // Only prevent default and scroll horizontally if:
        // 1. We're not at the right edge when scrolling right, OR
        // 2. We're not at the left edge when scrolling left
        if ((!isAtRightEnd || scrollingLeft) && (!isAtLeftEnd || scrollingRight)) {
          e.preventDefault();
          scrollRef.current.scrollLeft += e.deltaY;
        }
        // Otherwise, don't prevent default, allowing normal page scrolling
      }
    };
  
    const currentScrollRef = scrollRef.current;
    
    if (currentScrollRef) {
      currentScrollRef.addEventListener('wheel', handleWheel, { passive: false });
    }
    
    return () => {
      if (currentScrollRef) {
        currentScrollRef.removeEventListener('wheel', handleWheel);
      }
    };
  }, []);

  const features = [
    {
      title: "Personalized Recommendations",
      icon: <Sprout size={48} className="text-farmer-400" />,
      description: "Get AI-powered suggestions tailored to your specific farm conditions, crop preferences, and financial goals.",
      color: "from-gray-900 to-gray-800"
    },
    {
      title: "Market Analysis",
      icon: <BarChart size={48} className="text-farmer-400" />,
      description: "Access real-time data on market trends, crop pricing, and demand forecasts to make informed decisions.",
      color: "from-farmer-900 to-farmer-800"
    },
    {
      title: "Weather Integration",
      icon: <Cloud size={48} className="text-farmer-400" />,
      description: "Our system incorporates local weather data to optimize irrigation schedules and protect your crops.",
      color: "from-gray-900 to-gray-800"
    },
    {
      title: "Resource Optimization",
      icon: <Droplets size={48} className="text-farmer-400" />,
      description: "Minimize water usage and reduce fertilizer and pesticide application through precise recommendations.",
      color: "from-farmer-900 to-farmer-800"
    }
    // {
    //   title: "Sustainability Tracking",
    //   icon: <Leaf size={48} className="text-farmer-400" />,
    //   description: "Monitor your farm's environmental impact and progress toward sustainability goals over time.",
    //   color: "from-gray-900 to-gray-800"
    // },
    // {
    //   title: "Expert Insights",
    //   icon: <Tractor size={48} className="text-farmer-400" />,
    //   description: "Access agricultural expertise and best practices tailored to your specific farming context.",
    //   color: "from-farmer-900 to-farmer-800"
    // }
  ];

  return (
<section className="py-0 bg-black text-white w-full">
  <div className="px-4 py-10">
    <h2 className="text-4xl font-bold mb-2 ml-8">Our Features</h2>
    <p className="text-gray-300 text-lg ml-8">Scroll horizontally to explore our key capabilities</p>
  </div>
  
  <div 
    ref={scrollRef}
    className="flex overflow-x-auto w-full pb-10 pt-4 hide-scrollbar"
    style={{ 
      scrollbarWidth: 'none', 
      msOverflowStyle: 'none',
      WebkitOverflowScrolling: 'touch' 
    }}
  >
    <div className="flex space-x-6 pl-8 pr-16">
      {
       features.map((feature, index) => (
        <motion.div
          key={index}
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: index * 0.1 }}
          viewport={{ once: true }}
          whileHover={{ 
            y: -15, 
            boxShadow: "0 20px 25px -5px rgba(0, 0, 0, 0.2), 0 10px 10px -5px rgba(0, 0, 0, 0.1)",
            transition: { type: "spring", stiffness: 300, damping: 15 }
          }}
          className={`min-w-[320px] h-[420px] flex-shrink-0 rounded-2xl bg-gradient-to-br ${feature.color} p-8 flex flex-col justify-between border border-gray-800 hover:border-farmer-400 transition-all duration-300`}
        >
          <div>
            <div className="mb-6 transform transition-transform duration-300 hover:scale-110">
              {feature.icon}
            </div>
            <h3 className="text-3xl font-bold mb-4">{feature.title}</h3>
            <p className="text-gray-300 text-lg">{feature.description}</p>
          </div>
          <Button 
            variant="outline" 
            className="w-full mt-6 border-farmer-600 text-farmer-300 hover:bg-farmer-800 hover:text-white hover:border-farmer-300 transition-all duration-300 transform hover:scale-105"
          >
            Learn More
          </Button>
        </motion.div>
      ))}
    </div>
  </div>
  
  {/* Scroll indicators */}
  <div className="flex justify-center items-center gap-3 py-6 bg-black">
    <div className="w-12 h-1 bg-farmer-600 rounded-full"></div>
    <div className="w-3 h-3 bg-farmer-300 rounded-full animate-pulse"></div>
    <div className="w-3 h-3 bg-farmer-500 rounded-full"></div>
    <div className="w-3 h-3 bg-farmer-700 rounded-full"></div>
  </div>
</section>
  );
};

export default FeaturesSection;