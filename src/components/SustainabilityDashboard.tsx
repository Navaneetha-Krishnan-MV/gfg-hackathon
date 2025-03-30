import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import WeatherWidget from "./WeatherWidget";

const SustainabilityDashboard: React.FC = () => {
  const sustainabilityMetrics = [
    { name: "Water Usage", value: 68, target: 100, unit: "liters/ha", color: "bg-blue-500" },
    { name: "Carbon Footprint", value: 42, target: 100, unit: "kg CO₂/ton", color: "bg-farmer-600" },
    { name: "Biodiversity Score", value: 78, target: 100, unit: "points", color: "bg-farmer-500" },
    { name: "Soil Health", value: 85, target: 100, unit: "index", color: "bg-amber-500" }
  ];

  const practicesList = [
    { practice: "Crop Rotation", status: "Implemented", impact: "High" },
    { practice: "Cover Cropping", status: "In Progress", impact: "Medium" },
    { practice: "Reduced Tillage", status: "Implemented", impact: "High" },
    { practice: "Precision Irrigation", status: "Planned", impact: "High" }
  ];

  return (
    <div className="p-6 grid grid-cols-1 lg:grid-cols-3 gap-6">
      <div className="lg:col-span-1">
        <WeatherWidget />
        
        <Card className="mt-6 border-farmer-200 shadow-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-farmer-700 text-lg">Sustainable Practices</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {practicesList.map((item, index) => (
                <div key={index} className="flex justify-between items-center pb-2 border-b border-farmer-100 last:border-0">
                  <div>
                    <p className="font-medium text-farmer-800">{item.practice}</p>
                    <p className="text-sm text-farmer-600">{item.status}</p>
                  </div>
                  <span className={`px-2 py-1 rounded text-xs font-medium ${
                    item.impact === "High" 
                      ? "bg-farmer-100 text-farmer-800" 
                      : item.impact === "Medium" 
                        ? "bg-amber-100 text-amber-800" 
                        : "bg-blue-100 text-blue-800"
                  }`}>
                    {item.impact} Impact
                  </span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
      
      <div className="lg:col-span-2">
        <Card className="border-farmer-200 shadow-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-farmer-700 text-xl">Sustainability Metrics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              {sustainabilityMetrics.map((metric, index) => (
                <div key={index} className="space-y-2">
                  <div className="flex justify-between">
                    <h3 className="font-medium text-farmer-700">{metric.name}</h3>
                    <span className="text-farmer-600">
                      {metric.value}/{metric.target} {metric.unit}
                    </span>
                  </div>
                  <Progress value={(metric.value / metric.target) * 100} className="h-2" indicatorClassName={metric.color} />
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
          <Card className="border-farmer-200 shadow-sm">
            <CardHeader className="pb-2">
              <CardTitle className="text-farmer-700 text-lg">Certification Progress</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-farmer-700">Organic Certification</span>
                  <span className="bg-farmer-100 text-farmer-800 px-2 py-1 rounded text-xs">In Progress</span>
                </div>
                <Progress value={65} className="h-2" indicatorClassName="bg-farmer-500" />
                <p className="text-sm text-farmer-600">2 of 5 requirements completed</p>
                
                <div className="flex justify-between items-center mt-6">
                  <span className="text-farmer-700">Regenerative Agriculture</span>
                  <span className="bg-green-100 text-green-800 px-2 py-1 rounded text-xs">Certified</span>
                </div>
                <Progress value={100} className="h-2" indicatorClassName="bg-green-500" />
                <p className="text-sm text-farmer-600">Certification valid until Dec 2023</p>
              </div>
            </CardContent>
          </Card>
          
          <Card className="border-farmer-200 shadow-sm">
            <CardHeader className="pb-2">
              <CardTitle className="text-farmer-700 text-lg">Environmental Impact</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-center mb-4">
                <span className="text-3xl font-bold text-farmer-700">-32%</span>
                <p className="text-farmer-600">Carbon reduction from last year</p>
              </div>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-sm text-farmer-700">Water Conservation</span>
                  <span className="text-sm font-medium text-farmer-800">+24%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-farmer-700">Renewable Energy</span>
                  <span className="text-sm font-medium text-farmer-800">+18%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-farmer-700">Waste Reduction</span>
                  <span className="text-sm font-medium text-farmer-800">+15%</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default SustainabilityDashboard;