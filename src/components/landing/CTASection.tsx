import React from "react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Link } from "react-router-dom";

const CTASection = () => {
  return (
    <section className="py-20 bg-gradient-to-r from-farmer-900 to-farmer-700 text-white">
      <div className="container mx-auto px-4 text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
        >
          <h2 className="text-4xl font-bold mb-6">Ready to Transform Your Farm?</h2>
          <p className="text-xl max-w-3xl mx-auto mb-10 text-farmer-100">
            Join thousands of farmers who are improving yields, reducing costs, and farming more sustainably with FarmTech.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link to="/">
              <Button size="lg" className="bg-white text-farmer-800 hover:bg-farmer-100 transition-all transform hover:scale-105">
                Start Dashboard
              </Button>
            </Link>
            <Button size="lg" className="bg-white text-farmer-800 hover:bg-farmer-100 transition-all transform hover:scale-105">
              Contact Us
            </Button>
          </div>
        </motion.div>
      </div>
    </section>
  );
};

export default CTASection;