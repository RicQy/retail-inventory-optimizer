# Datetime Type Audit and Fix Tracking

## Overview
This document tracks all occurrences of datetime-related type specifications across the codebase that need to be updated for better type safety and compatibility.

## Search Results Summary
- **Total files containing datetime types**: 3
- **Total occurrences found**: 6

## Detailed Findings

### 1. `app/etl/config.py`
**Line 48**: 
```python
'date': 'datetime64[ns]',
```
**Context**: Part of `RETAIL_DATA_SCHEMA` configuration dictionary
**Intended Replacement**: `pd.Timestamp` or `datetime.datetime` (depending on usage context)
**Priority**: Medium - This is a configuration value, not a runtime type check

### 2. `app/etl/validation.py`
**Line 79**:
```python
pd.DatetimeDtype,
```
**Context**: Pandera Column definition in `RETAIL_SALES_SCHEMA`
**Intended Replacement**: `pd.Timestamp` or use Pandera's built-in datetime validation
**Priority**: High - Core validation logic

**Line 122**:
```python
"date": Column(pd.DatetimeDtype, nullable=False),
```
**Context**: Pandera Column definition in `PROCESSED_DATA_SCHEMA`
**Intended Replacement**: `pd.Timestamp` or use Pandera's built-in datetime validation
**Priority**: High - Core validation logic

**Line 199**:
```python
'date': 'datetime64[ns]',
```
**Context**: Expected dtypes dictionary in `calculate_data_quality_metrics` method
**Intended Replacement**: `pd.Timestamp` or string comparison with proper datetime format
**Priority**: Medium - Data quality checking

**Line 468**:
```python
'date': 'datetime64[ns]',
```
**Context**: Reference schema dtypes in `validate_data` method
**Intended Replacement**: `pd.Timestamp` or string comparison with proper datetime format
**Priority**: Medium - Schema drift detection

### 3. `tests/test_validation_system.py`
**Line 143**:
```python
'date': 'datetime64[ns]',
```
**Context**: Test reference schema dtypes
**Intended Replacement**: `pd.Timestamp` or string comparison with proper datetime format
**Priority**: Low - Test code, should mirror production changes

**Line 254**:
```python
assert transformed_data['date'].dtype == 'datetime64[ns]'
```
**Context**: Test assertion checking data type
**Intended Replacement**: Update assertion to match new datetime type specification
**Priority**: Low - Test code, should mirror production changes

## Implementation Plan

### Phase 1: Core Validation Logic (High Priority)
1. **File**: `app/etl/validation.py`
   - Lines 79, 122: Replace `pd.DatetimeDtype` with appropriate Pandera datetime validation
   - Consider using `pa.dtypes.Timestamp` or built-in datetime checks

### Phase 2: Data Quality and Schema Checking (Medium Priority)
1. **File**: `app/etl/validation.py`
   - Lines 199, 468: Replace hardcoded `'datetime64[ns]'` strings with proper type checking
2. **File**: `app/etl/config.py`
   - Line 48: Update configuration to use more explicit datetime type specification

### Phase 3: Test Updates (Low Priority)
1. **File**: `tests/test_validation_system.py`
   - Lines 143, 254: Update test expectations to match new datetime type handling

## Recommended Replacements

### For Pandera Schemas:
```python
# Current
Column(pd.DatetimeDtype, ...)

# Recommended
Column(pd.Timestamp, ...)
# OR
Column(pa.DateTime, ...)
```

### For Type String Comparisons:
```python
# Current
'date': 'datetime64[ns]'

# Recommended
'date': str(pd.Timestamp)
# OR use isinstance() checks instead of string comparison
```

### For Configuration:
```python
# Current
'date': 'datetime64[ns]'

# Recommended
'date': 'timestamp'  # More generic
# OR
'date': pd.Timestamp.__name__
```

## Notes
- The main issue is with `pd.DatetimeDtype` usage in Pandera schemas
- String-based datetime type comparisons should be replaced with proper type checking
- Test files should be updated last to match production code changes
- Consider using Pandera's built-in datetime validation instead of pandas-specific types

## Status
- [x] Audit completed
- [ ] Phase 1 implementation
- [ ] Phase 2 implementation  
- [ ] Phase 3 implementation
- [ ] Validation and testing
