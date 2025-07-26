# New Backlog Item Discovery Analysis

## Coverage Analysis Results
Based on current test coverage analysis, the following modules require attention:

### High-Priority Coverage Gaps
1. **API Module** (`src/lexgraph_legal_rag/api.py`): 21% coverage, 246 lines, 29 classes/functions
   - Core application API endpoints
   - Critical for application functionality
   - High impact if broken

2. **Document Pipeline** (`src/lexgraph_legal_rag/document_pipeline.py`): 12% coverage, 170 lines, 16 classes/functions  
   - Core document processing pipeline
   - Critical for search functionality
   - High impact, medium complexity

3. **Auth Module** (`src/lexgraph_legal_rag/auth.py`): 21% coverage, 116 lines
   - Already has comprehensive tests from previous work
   - Coverage may be reporting incorrectly due to test isolation

### Medium-Priority Coverage Gaps
4. **Config Module** (`src/lexgraph_legal_rag/config.py`): 18% coverage, 73 lines
   - Configuration management
   - Medium impact, low complexity

5. **Alerting Module** (`src/lexgraph_legal_rag/alerting.py`): 33% coverage, 141 lines, 30 classes/functions
   - Monitoring and alerting system
   - Medium impact for operations

## Recommended New Backlog Items

### Priority 1: Document Pipeline Test Coverage
- **WSJF Estimate**: ~9.0 (High value, high time criticality, high risk reduction, medium effort)
- **Rationale**: Core functionality, only 12% coverage, critical for search operations

### Priority 2: API Module Test Coverage  
- **WSJF Estimate**: ~8.0 (High value, medium time criticality, high risk reduction, high effort)
- **Rationale**: Main application interface, many endpoints to test

### Priority 3: Config Module Test Coverage
- **WSJF Estimate**: ~6.0 (Medium value, low time criticality, medium risk reduction, low effort)
- **Rationale**: Configuration is important but lower complexity

## Notes
- No TODO/FIXME comments found in codebase
- Current completion rate: 75% of backlog items
- FAISS index module successfully completed with 97% coverage