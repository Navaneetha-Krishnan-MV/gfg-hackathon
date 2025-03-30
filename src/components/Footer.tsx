import React from "react";

const Footer = () => {
  return (
    <footer className="bg-farmer-950 text-white py-12">
      <div className="container mx-auto px-4">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div>
            <h3 className="text-xl font-bold mb-4">FARMTECH</h3>
            <p className="text-farmer-300 mb-4">
              Transforming agriculture with AI-powered insights for sustainable farming.
            </p>
          </div>
          <div>
            <h4 className="font-semibold mb-4">Features</h4>
            <ul className="space-y-2 text-farmer-300">
              <li>Sustainability Dashboard</li>
              <li>Financial Insights</li>
              <li>Market Analysis</li>
              <li>Weather Integration</li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold mb-4">Resources</h4>
            <ul className="space-y-2 text-farmer-300">
              <li>Blog</li>
              <li>Help Center</li>
              <li>Case Studies</li>
              <li>Partner Program</li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold mb-4">Contact Us</h4>
            <ul className="space-y-2 text-farmer-300">
              <li>info@farmtech.com</li>
              <li>+1 (555) 123-4567</li>
              <li>123 Agriculture Way</li>
              <li>Farm City, FC 12345</li>
            </ul>
          </div>
        </div>
        <div className="border-t border-farmer-800 mt-8 pt-8 text-center text-farmer-400 text-sm">
          &copy; {new Date().getFullYear()} FarmTech. All rights reserved.
        </div>
      </div>
    </footer>
  );
};

export default Footer;