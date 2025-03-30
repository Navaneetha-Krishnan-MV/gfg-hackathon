import React from "react";
import CustomCursor from "./landing/CustomCursor";
import HeroSection from "./landing/HeroSection";
import MissionSection from "./landing/MissionSection";
import FeaturesSection from "./landing/FeaturesSection";
import TestimonialsSection from "./landing/TestimonialsSection";
import CTASection from "./landing/CTASection";

const LandingPage = () => {
  return (
    <div className="relative min-h-screen w-full bg-white overflow-x-hidden">
       <CustomCursor />
       <HeroSection />
       <MissionSection />
       <FeaturesSection />
       <TestimonialsSection />
       <CTASection />
    </div>
  );
};

export default LandingPage;