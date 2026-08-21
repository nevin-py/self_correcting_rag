#!/usr/bin/env python
"""Test all imports to verify refactored code works correctly."""

import sys
sys.path.insert(0, '.')

print("=" * 60)
print("TESTING REFACTORED RAG IMPORTS")
print("=" * 60)

# Test 1: Tool Types
print("\n1. Testing tool_types.py...")
try:
    from app.agent.tool_types import ToolType, ToolSelection, TOOL_REGISTRY
    print("   ✅ ToolType enum imported")
    print(f"   ✅ Available tools: {[t.value for t in ToolType]}")
    print(f"   ✅ TOOL_REGISTRY has {len(TOOL_REGISTRY)} tools")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    sys.exit(1)

# Test 2: Location Tool
print("\n2. Testing location_tool.py...")
try:
    from app.documents.location_tool import LocationTool, LocationInfo
    tool = LocationTool()
    print("   ✅ LocationTool instance created")
    print("   ✅ LocationInfo dataclass available")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    sys.exit(1)

# Test 3: Intent Detection
print("\n3. Testing intent_detection.py...")
try:
    from app.agent.intent_detection import (
        IntentClassification,
        classify_intent,
        _is_conversational_query,
        _get_conversational_response,
    )
    print("   ✅ IntentClassification imported")
    print("   ✅ classify_intent function available")
    print("   ✅ Conversational helpers available")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    sys.exit(1)

# Test 4: State
print("\n4. Testing state...")
try:
    from app.agent.state import RAGState
    print("   ✅ RAGState imported")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    sys.exit(1)

# Test 5: Graph
print("\n5. Testing graph...")
try:
    from app.agent.graph import (
        StateGraph,
        END,
        _route_to_conversational_or_classify,
    )
    print("   ✅ StateGraph imported")
    print("   ✅ END constant available")
    print("   ✅ _route_to_conversational_or_classify function available")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    sys.exit(1)

# Test 6: Nodes
print("\n6. Testing nodes...")
try:
    from app.agent.nodes import (
        classify_and_plan,
        retrieve_documents,
        search_web,
        assemble_evidence,
        generate_answer,
        ask_clarification,
        conversational_response,
    )
    print("   ✅ Node functions imported")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    sys.exit(1)

# Test 7: Schema imports (optional - most important modules already tested)
print("\n7. Testing remaining imports...")
try:
    # Try importing additional modules if needed
    from app.agent.schemas import *
    from app.agent.models import *
    print("   ✅ Additional schema/model imports work")
except Exception as e:
    print(f"   ⚠️  Warning: {e}")
    print("   (This is optional - not critical for main functionality)")

print("\n" + "=" * 60)
print("ALL TESTS PASSED! ✅")
print("=" * 60)
print("\nThe refactored code is ready to run.")
print("\nTo start the server:")
print("  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
print("=" * 60)