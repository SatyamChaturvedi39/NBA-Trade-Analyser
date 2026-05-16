import axios from 'axios';

// Use environment variable for production, fallback to localhost for development
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5000';

const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export const searchPlayers = async (query) => {
    try {
        const response = await api.get(`/api/search?q=${encodeURIComponent(query)}`);
        return response.data;
    } catch (error) {
        console.error("Error searching players:", error);
        throw error;
    }
};

export const getTeams = async () => {
    try {
        const response = await api.get('/api/teams');
        return response.data;
    } catch (error) {
        console.error("Error fetching teams:", error);
        throw error;
    }
};

export const getPlayersByTeam = async (teamCode) => {
    try {
        const response = await api.get(`/api/players?team=${encodeURIComponent(teamCode)}`);
        return response.data;
    } catch (error) {
        console.error("Error fetching roster:", error);
        throw error;
    }
};

export const analyzeTrade = async (teamA, teamB, sentA, sentB) => {
    try {
        const response = await api.post('/api/trade/evaluate', {
            team_a: teamA,
            team_b: teamB,
            sent_a: sentA,
            sent_b: sentB
        });
        return response.data;
    } catch (error) {
        console.error("Error analyzing trade:", error);
        throw error;
    }
};

export const predictPlayer = async (playerName) => {
    try {
        const response = await api.post('/api/predict', { player_name: playerName });
        return response.data;
    } catch (error) {
        console.error("Error predicting player stats:", error);
        throw error;
    }
};

export default api;
