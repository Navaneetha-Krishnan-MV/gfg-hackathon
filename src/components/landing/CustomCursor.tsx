import React, { useRef, useEffect } from "react";

const CustomCursor = () => {
  const cursorRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (cursorRef.current) {
        cursorRef.current.style.left = `${e.clientX}px`;
        cursorRef.current.style.top = `${e.clientY}px`;
      }
    };
    
    document.addEventListener("mousemove", handleMouseMove);
    
    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
    };
  }, []);

  return (
    <div 
      ref={cursorRef} 
      className="pointer-events-none fixed w-8 h-8 rounded-full bg-farmer-300/50 z-50 transform -translate-x-1/2 -translate-y-1/2 mix-blend-difference hidden md:block"
      style={{ transition: "transform 0.1s ease-out, left 0.1s linear, top 0.1s linear" }}
    />
  );
};

export default CustomCursor;