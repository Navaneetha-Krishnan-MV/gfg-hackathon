import React, { useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { LeafyGreen, DollarSign, House } from "lucide-react";

const Navbar: React.FC = () => {
  const location = useLocation();
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  return (
    <nav className="bg-farmer-600 text-white py-3 px-4 shadow-md">
      <div className="container mx-auto flex justify-between items-center">
        <div className="flex items-center">
          <Link to="/" className="text-2xl font-bold tracking-wider text-white hover:text-white">
            AGROSAGE
          </Link>
        </div>
        <button
          className="lg:hidden text-white focus:outline-none"
          onClick={() => setIsMenuOpen(!isMenuOpen)}
        >
          <svg
            className="w-6 h-6"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M4 6h16M4 12h16m-7 6h7"
            />
          </svg>
        </button>
        <div
          className={`${
            isMenuOpen ? "block" : "hidden"
          } lg:flex lg:space-x-2 flex-col lg:flex-row space-y-2 lg:space-y-0`}
        >
          <Link to="/">
            <Button
              variant={location.pathname === "/sustainability" ? "secondary" : "ghost"}
              className={`flex items-center border-0 gap-2 ${
                location.pathname === "/"
                  ? "bg-white text-farmer-700 border-none"
                  : "text-white bg-transparent hover:bg-farmer-500"
              } focus:ring-0 focus:outline-none`}
            >
              <House size={18} />
              <span>Home</span>
            </Button>
          </Link>
          <Link to="/sustainability">
            <Button
              variant={location.pathname === "/sustainability" ? "secondary" : "ghost"}
              className={`flex items-center border-0 gap-2 ${
                location.pathname === "/sustainability"
                  ? "bg-white text-farmer-700 border-none"
                  : "text-white bg-transparent hover:bg-farmer-500"
              } focus:ring-0 focus:outline-none`}
            >
              <LeafyGreen size={18} />
              <span>Sustainability</span>
            </Button>
          </Link>
          <Link to="/financial">
            <Button
              variant={location.pathname === "/financial" ? "secondary" : "ghost"}
              className={`flex items-center border-0 gap-2 ${
                location.pathname === "/financial"
                  ? "bg-white text-farmer-700 border-none"
                  : "text-white bg-transparent hover:bg-farmer-500"
              } focus:ring-0 focus:outline-none`}
            >
              <DollarSign size={18} />
              <span>Financial</span>
            </Button>
          </Link>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;