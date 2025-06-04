import asyncio
from typing import Any, Callable, Dict, Optional

class FunctionTool:
    """Wrapper for function tools."""
    
    def __init__(self, func: Callable, name: Optional[str] = None, description: Optional[str] = None):
        self.func = func
        self.name = name or func.__name__
        self.description = description or func.__doc__ or ""
        
    def to_invoke_tool(self) -> Dict[str, Any]:
        """Convert to invoke tool format."""
        # Get function signature for parameters
        import inspect
        sig = inspect.signature(self.func)
        
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            param_type = "string"  # Default type
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
                    
            properties[param_name] = {"type": param_type}
            
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
                
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
        
    async def call(self, **kwargs) -> Any:
        """Call the function with given arguments."""
        if asyncio.iscoroutinefunction(self.func):
            return await self.func(**kwargs)
        else:
            return self.func(**kwargs)


def function_tool(func: Callable) -> FunctionTool:
    """Decorator to create a function tool."""
    return FunctionTool(func)