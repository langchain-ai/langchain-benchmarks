"""A simple environment for evaluating an agent.

The purpose of this environment is to evaluate an agent's ability to use
the given tools to solve problems.

The environment contains fake data about users and their locations.

The agent should use the existing tools, and not try to access the data
directly.
"""
from typing import List, Callable

from langchain.tools import BaseTool
from langchain.tools import tool

USER_DATA = [
    {
        "id": "1",
        "name": "Alice",
        "email": "alice@gmail.com",
        "location": 1,  # Replaced location with an integer
        "favorite_color": "red",
        "favorite_foods": [1, 2, 3],  # Replace food names with IDs
    },
    {
        "id": "2",
        "name": "Bob",
        "email": "bob@hotmail.com",
        "location": 2,  # Replaced location with an integer
        "favorite_color": "orange",
        "favorite_foods": [4, 5, 6],  # Replace food names with IDs
    },
    {
        "id": "3",
        "name": "Charlie",
        "email": "charlie@yahoo.com",
        "location": 3,  # Replaced location with an integer
        "favorite_color": "yellow",
        "favorite_foods": [3, 7, 2],  # Replace food names with IDs
    },
    {
        "id": "4",
        "name": "Donna",
        "email": "donna@example.com",
        "location": 4,  # Replaced location with an integer
        "favorite_color": "green",
        "favorite_foods": [6, 1, 4],  # Replace food names with IDs
    },
    {
        "id": "5",
        "name": "Eve",
        "email": "eve@example.org",
        "location": 5,  # Replaced location with an integer
        "favorite_color": "blue",
        "favorite_foods": [5, 7, 4],  # Replace food names with IDs
    },
]

# Create a list of JSON data for locations with "current_weather" as a single string
LOCATION_DATA = [
    {
        "id": 1,
        "city": "New York",
        "current_time": "2023-11-14 10:30 AM",
        "current_weather": "Partly Cloudy, Temperature: 68°F",  # Example weather string
    },
    {
        "id": 2,
        "city": "Los Angeles",
        "current_time": "2023-11-14 7:45 AM",
        "current_weather": "Sunny, Temperature: 75°F",  # Example weather string
    },
    {
        "id": 3,
        "city": "Chicago",
        "current_time": "2023-11-14 11:15 AM",
        "current_weather": "Mostly Cloudy, Temperature: 60°F",  # Example weather string
    },
    {
        "id": 4,
        "city": "Houston",
        "current_time": "2023-11-14 12:00 PM",
        "current_weather": "Rainy, Temperature: 55°F",  # Example weather string
    },
    {
        "id": 5,
        "city": "Miami",
        "current_time": "2023-11-14 1:20 PM",
        "current_weather": "Partly Cloudy, Temperature: 80°F",  # Example weather string
    },
]

FOOD_DATA = [
    {
        "id": 1,
        "name": "Pizza",
        "calories": 285,  # Calories per serving
        "serving_weight": 150,  # Approximate weight in grams for one serving
        "allergic_ingredients": ["Gluten", "Dairy"],
    },
    {
        "id": 2,
        "name": "Chocolate",
        "calories": 50,  # Calories per serving
        "serving_weight": 20,  # Approximate weight in grams for one serving
        "allergic_ingredients": ["Milk", "Soy"],
    },
    {
        "id": 3,
        "name": "Sushi",
        "calories": 300,  # Calories per serving
        "serving_weight": 200,  # Approximate weight in grams for one serving
        "allergic_ingredients": ["Fish", "Soy"],
    },
    {
        "id": 4,
        "name": "Burger",
        "calories": 350,  # Calories per serving
        "serving_weight": 200,  # Approximate weight in grams for one serving
        "allergic_ingredients": ["Gluten", "Dairy"],
    },
    {
        "id": 5,
        "name": "Ice Cream",
        "calories": 200,  # Calories per serving
        "serving_weight": 100,  # Approximate weight in grams for one serving
        "allergic_ingredients": ["Dairy"],
    },
    {
        "id": 6,
        "name": "Pasta",
        "calories": 180,  # Calories per serving
        "serving_weight": 100,  # Approximate weight in grams for one serving
        "allergic_ingredients": ["Gluten"],
    },
    {
        "id": 7,
        "name": "Salad",
        "calories": 50,  # Calories per serving
        "serving_weight": 100,  # Approximate weight in grams for one serving
        "allergic_ingredients": [],
    },
]


def _find_user(id: str) -> dict:
    """Find the user with the given user ID.

    Args:
        id: The user's ID.

    Returns:
        The user's data.
    """
    for user in USER_DATA:
        if user["id"] == id:
            return user
    raise ValueError(f"User ID {id} cannot be resolved")


def _find_location(id: str) -> dict:
    """Find the location with the given location ID.

    Args:
        id: The location's ID.

    Returns:
        The location's data.
    """
    for location in LOCATION_DATA:
        if location["id"] == id:
            return location
    raise ValueError(f"Location ID {id} cannot be resolved")


def _find_food(food_id: int) -> dict:
    """Find the food with the given food ID.

    Args:
        food_id: The food's ID.

    Returns:
        The food's data.
    """
    for food in FOOD_DATA:
        if food["id"] == food_id:
            return food
    raise ValueError(f"Food ID {food_id} cannot be resolved")


def get_available_functions() -> List[Callable]:
    """Get all the available functions."""

    def get_user_name(user_id: str) -> str:
        """Get the name of the user with the given user ID.

        Args:
            user_id: The user's ID.

        Returns:
            The user's name.
        """
        return _find_user(user_id)["name"]

    def get_user_email(user_id: str) -> str:
        """Get the email of the user with the given user ID.

        Args:
            user_id: The user's ID.

        Returns:
            The user's email.
        """
        return _find_user(user_id)["email"]

    def get_user_location(user_id: str) -> int:
        """Get the location ID of the user with the given user ID.

        Args:
            user_id: The user's ID.

        Returns:
            The user's location ID.
        """
        return _find_user(user_id)["location"]

    def get_user_favorite_color(user_id: str) -> str:
        """Get the favorite color of the user with the given user ID.

        Args:
            user_id: The user's ID.

        Returns:
            The user's favorite color.
        """
        return _find_user(user_id)["favorite_color"]

    def get_user_favorite_foods(user_id: str) -> List[int]:
        """Get the list of favorite foods of the user with the given user ID.

        Args:
            user_id: The user's ID.

        Returns:
            The list of favorite foods.
        """
        return _find_user(user_id)["favorite_foods"]

    def get_weather_at_location(location_id: int) -> str:
        """Get the current weather at the location with the given location ID.

        Args:
            location_id: The location's ID.

        Returns:
            The current weather at the location.
        """
        for location in LOCATION_DATA:
            if location["location_id"] == location_id:
                return location["current_weather"]
        raise ValueError("Invalid location ID")

    def get_city_for_location(location_id: int) -> str:
        """Get the city for the location with the given location ID.

        Args:
            location_id: The location's ID.

        Returns:
            The city name for the location.
        """
        return _find_location(location_id)["city"]

    def get_current_time_for_location(location_id: int) -> str:
        """Get the current time for the location with the given location ID.

        Args:
            location_id: The location's ID.

        Returns:
            The current time for the location.
        """
        return _find_location(location_id)["current_time"]

    def get_current_weather_for_location(location_id: int) -> str:
        """Get the current weather for the location with the given location ID.

        Args:
            location_id: The location's ID.

        Returns:
            The current weather for the location.
        """
        return _find_location(location_id)["current_weather"]

    def get_food_name(food_id: int) -> str:
        """Get the name of the food with the given food ID.

        Args:
            food_id: The food's ID.

        Returns:
            The name of the food.
        """
        return _find_food(food_id)["name"]

    def get_food_calories(food_id: int) -> int:
        """Get the calories per serving for the food with the given food ID.

        Args:
            food_id: The food's ID.

        Returns:
            The calories per serving of the food.
        """
        return _find_food(food_id)["calories"]

    def get_food_serving_weight(food_id: int) -> int:
        """Get the approximate serving weight in grams for the food with the given food ID.

        Args:
            food_id: The food's ID.

        Returns:
            The approximate serving weight in grams.
        """ # noqa: E501
        return _find_food(food_id)["serving_weight"]

    def get_food_allergic_ingredients(food_id: int) -> List[str]:
        """Get the list of allergic ingredients for the food with the given food ID.

        Args:
            food_id: The food's ID.

        Returns:
            The list of allergic ingredients.
        """
        return _find_food(food_id)["allergic_ingredients"]

    # Get all the functions defined in the scope of this function
    functions = [f for f in locals().values() if callable(f)]
    return functions


def get_tools() -> List[BaseTool]:
    """Get all the available tools."""
    functions = get_available_functions()
    return [tool(f) for f in functions]
