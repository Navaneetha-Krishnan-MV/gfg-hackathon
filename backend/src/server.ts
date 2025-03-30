import express, { Request, Response, RequestHandler } from "express";
import axios from "axios";
import dotenv from "dotenv";
import cors from "cors";

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;
const WEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather";
const API_KEY = process.env.WEATHER_API_KEY || "";
app.use(cors());

interface WeatherQuery {
  lat?: string;
  lon?: string;
}

// Weather API Endpoint
const weatherHandler: RequestHandler = async (req, res) => {
  try {
    const { lat, lon } = req.query;

    if (!lat || !lon) {
      res.status(400).json({ error: "Latitude and Longitude are required" });
      return;
    }

    const response = await axios.get(WEATHER_API_URL, {
      params: {
        lat,
        lon,
        appid: API_KEY,
        units: "metric",
      },
    });

    const weatherData = response.data;
    res.json({
      temperature: weatherData.main.temp,
      humidity: weatherData.main.humidity,
      windSpeed: weatherData.wind.speed,
      description: weatherData.weather[0].description,
      city: weatherData.name,
    });
  } catch (error) {
    console.error("Error fetching weather data:", error);
    res.status(500).json({ error: "Failed to fetch weather data" });
  }
};

app.get("/weather", weatherHandler);

// Start the server
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
