import React, { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Sun, Cloud, CloudRain } from "lucide-react";

const WeatherWidget: React.FC = () => {
  const [weather, setWeather] = useState({
    temp: null,
    condition: "",
    humidity: null,
    windSpeed: null,
    city: "Loading...",
  });

  const fetchWeatherData = async (latitude: number, longitude: number) => {
    try {
      const response = await fetch(`http://localhost:5000/weather?lat=${latitude}&lon=${longitude}`);
      const data = await response.json();
      setWeather({
        temp: data.temperature,
        condition: data.description,
        humidity: data.humidity,
        windSpeed: data.windSpeed,
        city: data.city,
      });
    } catch (error) {
      console.error("Error fetching weather data:", error);
    }
  };

  const getIPLocation = async () => {
    try {
      const response = await fetch("https://ipapi.co/json/");
      const data = await response.json();
      console.log("Fallback IP Location:", data);
      if (data.latitude && data.longitude) {
        fetchWeatherData(data.latitude, data.longitude);
      }
    } catch (error) {
      console.error("Error fetching IP location:", error);
    }
  };

  useEffect(() => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const { latitude, longitude } = position.coords;
          console.log("Latitude:", latitude, "Longitude:", longitude);
          fetchWeatherData(latitude, longitude);
        },
        (error) => {
          console.error("Geolocation error:", error);
          // Fallback to IP-based location on error
          getIPLocation();
        },
        { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
      );
    } else {
      // Fallback to IP-based location if Geolocation is not supported
      getIPLocation();
    }
  }, []);

  const getWeatherIcon = () => {
    if (weather.condition.includes("clear")) return <Sun className="h-10 w-10 text-yellow-500" />;
    if (weather.condition.includes("cloud")) return <Cloud className="h-10 w-10 text-gray-400" />;
    if (weather.condition.includes("rain")) return <CloudRain className="h-10 w-10 text-blue-400" />;
    return <Sun className="h-10 w-10 text-yellow-500" />;
  };

  return (
    <Card className="border-gray-200 shadow-sm bg-white">
      <CardHeader className="pb-2">
        <CardTitle className="text-gray-700 text-lg">Current Weather in {weather.city}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            {getWeatherIcon()}
            <div className="ml-4">
              <p className="text-3xl font-bold text-gray-800">{weather.temp ?? "--"}°C</p>
              <p className="text-gray-600 capitalize">{weather.condition || "Loading..."}</p>
            </div>
          </div>
          <div className="text-right">
            <p className="text-sm text-gray-700">Humidity: {weather.humidity ?? "--"}%</p>
            <p className="text-sm text-gray-700">Wind: {weather.windSpeed ?? "--"} km/h</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default WeatherWidget;
