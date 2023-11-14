"""A simple environment for evaluating an agent.

A simple environment to evaluate an agent's ability to use
a set of given tools to reference questions.

The environment contains fake data about users and their locations
and favorite foods.

The environment defines a set of tools that the agent can use to
access the data.

Agent performance should be evaluated solely based on the agent's
ability to use the tools to reference questions.
"""
from typing import List, Callable
from typing import TypedDict

from langchain.tools import BaseTool
from langchain.tools import tool

USER_DATA = [
    {
        "id": 1,
        "name": "Alice",
        "email": "alice@gmail.com",
        "location": 1,
        "favorite_color": "red",
        "favorite_foods": [1, 2, 3],
    },
    {
        "id": 2,
        "name": "Bob",
        "email": "bob@hotmail.com",
        "location": 2,
        "favorite_color": "orange",
        "favorite_foods": [4, 5, 6],
    },
    {
        "id": 3,
        "name": "Charlie",
        "email": "charlie@yahoo.com",
        "location": 3,
        "favorite_color": "yellow",
        "favorite_foods": [3, 7, 2],
    },
    {
        "id": 4,
        "name": "Donna",
        "email": "donna@example.com",
        "location": 4,
        "favorite_color": "green",
        "favorite_foods": [6, 1, 4],
    },
    {
        "id": 5,
        "name": "Eve",
        "email": "eve@example.org",
        "location": 5,
        "favorite_color": "blue",
        "favorite_foods": [5, 7, 4],
    },
    {
        "id": 6,
        "name": "Frank The Cat",
        "email": "frank.the.cat@langchain.dev",
        "location": 5,
        "favorite_color": "yellow",
        "favorite_foods": [3],
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
        "allergic_ingredients": ["Gluten", "Dairy"],
    },
    {
        "id": 2,
        "name": "Chocolate",
        "calories": 50,  # Calories per serving
        "allergic_ingredients": ["Milk", "Soy"],
    },
    {
        "id": 3,
        "name": "Sushi",
        "calories": 300,  # Calories per serving
        "allergic_ingredients": ["Fish", "Soy"],
    },
    {
        "id": 4,
        "name": "Burger",
        "calories": 350,  # Calories per serving
        "allergic_ingredients": ["Gluten", "Dairy"],
    },
    {
        "id": 5,
        "name": "Ice Cream",
        "calories": 200,  # Calories per serving
        "allergic_ingredients": ["Dairy"],
    },
    {
        "id": 6,
        "name": "Pasta",
        "calories": 180,  # Calories per serving
        "allergic_ingredients": ["Gluten"],
    },
    {
        "id": 7,
        "name": "Salad",
        "calories": 50,  # Calories per serving
        "allergic_ingredients": [],
    },
]


class SearchHit(TypedDict):
    """A search hit."""

    id: str
    value: str


def _similarity_search(data: List[dict], query: str, key: str) -> List[SearchHit]:
    """Return a list of data that matches the given query.

    Similarity score is jaccard similarity based on the number of shared
    characters between the query and the data.

    Args:
        data: The data to search.
        query: The query to search for.
        key: The key to search in.

    Returns:
        The list of matching data.
    """
    score_function = lambda x: len(set(x) & set(query)) / len(set(x) | set(query))
    re_ranked_data = sorted(data, key=lambda x: score_function(x[key]), reverse=True)
    return [{"id": d["id"], key: d[key]} for d in re_ranked_data]


def _get_user(id: int) -> dict:
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


def _get_location(id: int) -> dict:
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


def _get_food(food_id: int) -> dict:
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

    def get_user_name(user_id: int) -> str:
        """Get the name of the user with the given user ID.

        Args:
            user_id: The user's ID.

        Returns:
            The user's name.
        """
        return _get_user(user_id)["name"]

    def list_user_ids() -> List[str]:
        """List all the user IDs."""
        return [user["id"] for user in USER_DATA]

    def find_users_by_name(name: str) -> List[SearchHit]:
        """Find users with the given name.

        Args:
            name: The name to search for.

        Returns:
            The list of matching users.
        """
        return _similarity_search(USER_DATA, name, "name")

    def find_locations_by_name(city: str) -> List[SearchHit]:
        """Find locations with the given city name."""
        return _similarity_search(LOCATION_DATA, city, "city")

    def find_foods_by_name(food: str) -> List[SearchHit]:
        """Find foods with the given name."""
        return _similarity_search(FOOD_DATA, food, "name")

    def get_user_email(user_id: int) -> str:
        """Get the email of the user with the given user ID.

        Args:
            user_id: The user's ID.

        Returns:
            The user's email.
        """
        return _get_user(user_id)["email"]

    def get_user_location(user_id: int) -> int:
        """Get the location ID of the user with the given user ID.

        Args:
            user_id: The user's ID.

        Returns:
            The user's location ID.
        """
        return _get_user(user_id)["location"]

    def get_user_favorite_color(user_id: int) -> str:
        """Get the favorite color of the user with the given user ID.

        Args:
            user_id: The user's ID.

        Returns:
            The user's favorite color.
        """
        return _get_user(user_id)["favorite_color"]

    def list_user_ids() -> List[int]:
        """List all the user IDs."""
        return [user["id"] for user in USER_DATA]

    def get_user_favorite_foods(user_id: int) -> List[int]:
        """Get the list of favorite foods of the user with the given user ID.

        Args:
            user_id: The user's ID.

        Returns:
            The list of favorite foods.
        """
        return _get_user(user_id)["favorite_foods"]

    def get_weather_at_location(location_id: int) -> str:
        """Get the current weather at the location with the given location ID.

        Args:
            location_id: The location's ID.

        Returns:
            The current weather at the location.
        """
        return _get_location(location_id)["current_weather"]

    def get_city_for_location(location_id: int) -> str:
        """Get the city for the location with the given location ID.

        Args:
            location_id: The location's ID.

        Returns:
            The city name for the location.
        """
        return _get_location(location_id)["city"]

    def get_current_time_for_location(location_id: int) -> str:
        """Get the current time for the location with the given location ID.

        Args:
            location_id: The location's ID.

        Returns:
            The current time for the location.
        """
        return _get_location(location_id)["current_time"]

    def get_current_weather_for_location(location_id: int) -> str:
        """Get the current weather for the location with the given location ID.

        Args:
            location_id: The location's ID.

        Returns:
            The current weather for the location.
        """
        return _get_location(location_id)["current_weather"]

    def get_food_name(food_id: int) -> str:
        """Get the name of the food with the given food ID.

        Args:
            food_id: The food's ID.

        Returns:
            The name of the food.
        """
        return _get_food(food_id)["name"]

    def get_food_calories(food_id: int) -> int:
        """Get the calories per serving for the food with the given food ID.

        Args:
            food_id: The food's ID.

        Returns:
            The calories per serving of the food.
        """
        return _get_food(food_id)["calories"]

    def get_food_allergic_ingredients(food_id: int) -> List[str]:
        """Get the list of allergic ingredients for the food with the given food ID.

        Args:
            food_id: The food's ID.

        Returns:
            The list of allergic ingredients.
        """
        return _get_food(food_id)["allergic_ingredients"]

    def get_current_user_id() -> int:
        """Get the current user's ID.

        Returns:
            The current user's ID.
        """
        return 3

    # Get all the functions defined in the scope of this function
    functions = [f for f in locals().values() if callable(f)]
    return functions


def get_tools() -> List[BaseTool]:
    """Get all the available tools."""
    functions = get_available_functions()
    return [tool(f) for f in functions]


DATASET = [
    # 1-step questions
    {
        "question": "What is the city for location ID 1?",
        "reference": "New York",
        "expected_steps": ["get_city_for_location"],
    },
    {
        "question": "What is the name of Food ID 6?",
        "reference": "Pasta",
        "expected_steps": ["get_food_name"],
    },
    {
        "question": "what is eve's user id?",
        "reference": "5",
        "expected_steps": ["find_users_by_name"],
    },
    {
        "question": "get the current user id",
        "reference": "3",
        "expected_steps": ["get_current_user_id"],
    },
    # 1-step + counting
    {
        "question": "How many users by the name of bob?",
        "reference": "1",
        "expected_steps": ["find_users_by_name"],
    },
    # 2-step questions
    {
        "question": "what is alice's email address?",
        "reference": "alice@gmail.com",
        "expected_steps": ["find_users_by_name", "get_user_email"],
    },
    {
        "question": "find donna's favorite color",
        "reference": "green",
        "expected_steps": ["find_users_by_name", "get_user_favorite_color"],
    },
    {
        "question": "weather in LA right now?",
        "reference": "Sunny, Temperature: 75°F",
        "expected_steps": [
            "find_locations_by_name",
            "get_current_weather_for_location",
        ],
    },
    {
        "question": "time in chicago",
        "reference": "2023-11-14 11:15 AM",
        "expected_steps": ["find_locations_by_name", "get_current_time_for_location"],
    },
    {
        "question": "list the allergens in chocolate",
        "reference": "milk, soy",
        "expected_steps": ["find_foods_by_name", "get_food_allergic_ingredients"],
    },
    {
        "question": "If i eat a serving of pizza, how many calories will I consume?",
        "reference": "285 calories",
        "expected_steps": ["find_foods_by_name", "get_food_calories"],
    },
    # 2-step has irrelevant information
    {
        "question": "eve ate a serving of sushi, what allergens was she exposed to?",
        "reference": "fish, soy",
        "expected_steps": [
            "find_foods_by_name",
            "get_food_allergic_ingredients",
        ],
    },
    {
        "question": "Frank who is Even's friend is allergic to dairy. "
        "Can he eat the salad?",
        "reference": "yes",
        "expected_steps": [
            "find_users_by_name",
            "get_food_allergic_ingredients",
        ],
    },
    # 3-step questions
    {
        "question": "do bob and alice live in the same city?",
        "reference": "no",
        "expected_steps": [
            "find_users_by_name",
            # Here the ordering of the steps can be different
            "get_user_location",
            "get_city_for_location",
            "get_user_location",
            "get_city_for_location",
        ],
    },
    {
        "question": "whats the name of the city where bob lives?",
        "reference": "Los Angeles",
        "expected_steps": [
            "find_users_by_name",
            "get_user_location",
            "get_city_for_location",
        ],
    },
    {
        "question": "Donna is about to go outside. Does she need an umbrella?",
        "reference": "yes",
        "expected_steps": [
            "find_users_by_name",
            "get_user_location",
            "get_current_weather_for_location",
        ],
    },
    {
        "question": "Is it likely that Donna is awake right now?",
        "reference": "yes",
        "expected_steps": [
            "find_users_by_name",
            "get_user_location",
            "get_current_time_for_location",
        ],
    },
    {
        "question": "do alice and charlie use the same email provider?",
        "reference": "no",
        "expected_steps": [
            "find_users_by_name",
            "get_user_email",
            "get_user_email",
        ],
    },
    # 4-step questions
    {
        "question": "Is it likely that Donna is outside with an umbrella at this time?",
        "reference": "yes",
        "expected_steps": [
            "find_users_by_name",
            "get_user_location",
            # Order of steps can be different
            "get_current_time_for_location",
            "get_current_weather_for_location",
        ],
    },
    # Many steps
    {
        "question": "Which users live in the same city as Eve?",
        "reference": "Frank The Cat",
        "expected_steps": [
            "list_user_ids",
            # Impossible to tell which order will be used
            "get_user_location",
            "get_user_location",
            "get_user_location",
            "get_user_location",
            "get_user_location",
            "get_user_location",
        ],
    },
]
