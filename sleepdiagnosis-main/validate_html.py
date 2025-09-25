#!/usr/bin/env python3
"""
Simple HTML validation script to check for basic syntax errors
"""

import re
import sys

def validate_html_file(file_path):
    """Validate HTML file for basic syntax errors"""
    errors = []
    warnings = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        errors.append(f"Cannot read file: {e}")
        return errors, warnings
    
    # Check for basic HTML structure
    if not re.search(r'<html[^>]*>', content, re.IGNORECASE):
        errors.append("Missing <html> tag")
    
    if not re.search(r'<head[^>]*>', content, re.IGNORECASE):
        errors.append("Missing <head> tag")
    
    if not re.search(r'<body[^>]*>', content, re.IGNORECASE):
        errors.append("Missing <body> tag")
    
    # Check for unclosed script tags
    script_opens = len(re.findall(r'<script[^>]*>', content, re.IGNORECASE))
    script_closes = len(re.findall(r'</script>', content, re.IGNORECASE))
    
    if script_opens != script_closes:
        errors.append(f"Unmatched script tags: {script_opens} opening, {script_closes} closing")
    
    # Check for basic JavaScript syntax issues in script blocks
    script_blocks = re.findall(r'<script[^>]*>(.*?)</script>', content, re.DOTALL | re.IGNORECASE)
    
    for i, script in enumerate(script_blocks):
        # Check for basic syntax issues
        if '}}' in script:
            # Check for potential template literal issues
            template_literals = re.findall(r'`[^`]*`', script)
            for literal in template_literals:
                if literal.count('{') != literal.count('}'):
                    errors.append(f"Script block {i+1}: Unmatched braces in template literal")
        
        # Check for unmatched parentheses in function calls
        paren_count = script.count('(') - script.count(')')
        if paren_count != 0:
            warnings.append(f"Script block {i+1}: Unmatched parentheses (diff: {paren_count})")
        
        # Check for unmatched curly braces
        brace_count = script.count('{') - script.count('}')
        if brace_count != 0:
            warnings.append(f"Script block {i+1}: Unmatched curly braces (diff: {brace_count})")
    
    # Check for common HTML issues
    if re.search(r'<[^>]*\s+[^>=]+=[^"\']\s*[^>]*>', content):
        warnings.append("Possible unquoted attribute values")
    
    # Check for potential encoding issues
    if '�' in content:
        warnings.append("Possible encoding issues detected")
    
    return errors, warnings

def main():
    file_path = 'c:/Users/Akarsh PR/Downloads/SleepApnea/index.html'
    
    print("Validating HTML file...")
    print("=" * 50)
    
    errors, warnings = validate_html_file(file_path)
    
    if not errors and not warnings:
        print("✅ No issues found!")
    else:
        if errors:
            print("❌ ERRORS:")
            for error in errors:
                print(f"  - {error}")
        
        if warnings:
            print("⚠️  WARNINGS:")
            for warning in warnings:
                print(f"  - {warning}")
    
    print("\n" + "=" * 50)
    print("Validation complete!")

if __name__ == "__main__":
    main()