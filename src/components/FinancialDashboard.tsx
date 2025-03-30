import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import WeatherWidget from "./WeatherWidget";

const FinancialDashboard: React.FC = () => {
  const revenueData = [
    { name: "Jan", revenue: 4000 },
    { name: "Feb", revenue: 3000 },
    { name: "Mar", revenue: 5000 },
    { name: "Apr", revenue: 2780 },
    { name: "May", revenue: 5890 },
    { name: "Jun", revenue: 6390 },
    { name: "Jul", revenue: 3490 },
  ];
  
  const crops = [
    { name: "Corn", revenue: 42500, cost: 18300, profit: 24200, profitMargin: 57 },
    { name: "Soybeans", revenue: 38700, cost: 15800, profit: 22900, profitMargin: 59 },
    { name: "Wheat", revenue: 27300, cost: 14200, profit: 13100, profitMargin: 48 },
  ];
  
  const expenses = [
    { category: "Seeds", amount: 12500, percentage: 25 },
    { category: "Fertilizer", amount: 9800, percentage: 19 },
    { category: "Labor", amount: 8700, percentage: 17 },
    { category: "Equipment", amount: 7500, percentage: 15 },
    { category: "Other", amount: 12000, percentage: 24 },
  ];

  return (
    <div className="p-6 grid grid-cols-1 lg:grid-cols-3 gap-6">
      <div className="lg:col-span-1">
        <WeatherWidget />
        
        <Card className="mt-6 border-farmer-200 shadow-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-farmer-700 text-lg">Expense Breakdown</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {expenses.map((expense, index) => (
                <div key={index} className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-farmer-700">{expense.category}</span>
                    <span className="text-farmer-800 font-medium">${expense.amount.toLocaleString()}</span>
                  </div>
                  <Progress value={expense.percentage} className="h-2" indicatorClassName="bg-farmer-600" />
                  <p className="text-xs text-farmer-500 text-right">{expense.percentage}% of total</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
      
      <div className="lg:col-span-2">
        <Card className="border-farmer-200 shadow-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-farmer-700 text-xl">Revenue Overview</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart
                  data={revenueData}
                  margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#edf2f7" />
                  <XAxis dataKey="name" stroke="#4a5568" />
                  <YAxis stroke="#4a5568" />
                  <Tooltip />
                  <Area 
                    type="monotone" 
                    dataKey="revenue" 
                    stroke="#16a34a" 
                    fill="#dcfce7" 
                    fillOpacity={0.8} 
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
        
        <Card className="border-farmer-200 shadow-sm mt-6">
          <CardHeader className="pb-2">
            <CardTitle className="text-farmer-700 text-lg">Crop Profitability</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full border-collapse">
                <thead>
                  <tr className="border-b border-farmer-200 text-left">
                    <th className="py-3 px-4 text-farmer-700">Crop</th>
                    <th className="py-3 px-4 text-farmer-700 text-right">Revenue</th>
                    <th className="py-3 px-4 text-farmer-700 text-right">Cost</th>
                    <th className="py-3 px-4 text-farmer-700 text-right">Profit</th>
                    <th className="py-3 px-4 text-farmer-700 text-right">Margin</th>
                  </tr>
                </thead>
                <tbody>
                  {crops.map((crop, index) => (
                    <tr key={index} className="border-b border-farmer-100 last:border-0">
                      <td className="py-3 px-4 font-medium text-farmer-800">{crop.name}</td>
                      <td className="py-3 px-4 text-farmer-600 text-right">${crop.revenue.toLocaleString()}</td>
                      <td className="py-3 px-4 text-farmer-600 text-right">${crop.cost.toLocaleString()}</td>
                      <td className="py-3 px-4 text-farmer-700 font-medium text-right">${crop.profit.toLocaleString()}</td>
                      <td className="py-3 px-4 text-right">
                        <span className={`px-2 py-1 rounded text-xs font-medium ${
                          crop.profitMargin > 50 
                            ? "bg-farmer-100 text-farmer-800" 
                            : "bg-amber-100 text-amber-800"
                        }`}>
                          {crop.profitMargin}%
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
                <tfoot>
                  <tr className="bg-farmer-50">
                    <td className="py-3 px-4 font-medium text-farmer-800">Total</td>
                    <td className="py-3 px-4 text-farmer-800 font-medium text-right">
                      ${crops.reduce((sum, crop) => sum + crop.revenue, 0).toLocaleString()}
                    </td>
                    <td className="py-3 px-4 text-farmer-800 font-medium text-right">
                      ${crops.reduce((sum, crop) => sum + crop.cost, 0).toLocaleString()}
                    </td>
                    <td className="py-3 px-4 text-farmer-800 font-medium text-right">
                      ${crops.reduce((sum, crop) => sum + crop.profit, 0).toLocaleString()}
                    </td>
                    <td className="py-3 px-4 text-right">
                      <span className="px-2 py-1 rounded text-xs font-medium bg-farmer-100 text-farmer-800">
                        {Math.round(crops.reduce((sum, crop) => sum + crop.profit, 0) / 
                        crops.reduce((sum, crop) => sum + crop.revenue, 0) * 100)}%
                      </span>
                    </td>
                  </tr>
                </tfoot>
              </table>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default FinancialDashboard;