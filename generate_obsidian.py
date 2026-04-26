import json
import os

def main():
    graph_path = r'd:\1_ai-model\graphify-out\graph.json'
    out_dir = r'd:\1_ai-model\graphify-out\obsidian'
    
    if not os.path.exists(graph_path):
        print(f"Graph file not found: {graph_path}")
        return

    with open(graph_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    nodes = data.get('nodes', [])
    links = data.get('links', [])
    
    node_by_id = {n.get('id'): n for n in nodes}
    
    # Identify files, functions, and rationales
    id_to_file = {}
    file_nodes = {}
    function_nodes = {} # file -> list of functions
    rationale_links = {} # entity_id -> list of rationale strings
    
    # Parse rationale links
    for link in links:
        if link.get('relation') == 'rationale_for':
            src = link.get('source')
            tgt = link.get('target')
            
            # Looking at graph.json: 
            # source = the code entity (function or file)
            # target = the rationale node
            entity_id = src
            rationale_id = tgt
            
            # Check if source and target are swapped in some links?
            # E.g. {"_src": "structure_agent_rationale_1", "_tgt": "resume_scanner_backend_structure_agent_py", "source": "resume_scanner_backend_structure_agent_py", "target": "structure_agent_rationale_1"}
            # The "source" property points to the entity, the "target" property points to the rationale!
            # Wait, let's also check _src and _tgt just in case! 
            # In some cases: `source` is entity, `target` is rationale.
            # Let's check which one is the rationale.
            
            node_src = node_by_id.get(src)
            node_tgt = node_by_id.get(tgt)
            
            rationale_node = None
            target_entity_id = None
            
            if node_tgt and node_tgt.get('file_type') == 'rationale':
                rationale_node = node_tgt
                target_entity_id = src
            elif node_src and node_src.get('file_type') == 'rationale':
                rationale_node = node_src
                target_entity_id = tgt
                
            if rationale_node and target_entity_id:
                if target_entity_id not in rationale_links:
                    rationale_links[target_entity_id] = []
                cleaned_label = rationale_node.get('label', '').strip()
                if cleaned_label:
                    rationale_links[target_entity_id].append(cleaned_label)

    for node in nodes:
        node_id = node.get('id')
        source_file = node.get('source_file')
        file_type = node.get('file_type')
        
        if source_file:
            basename = os.path.basename(source_file)
            id_to_file[node_id] = basename
            
            # Is it a file or a function?
            if file_type == 'rationale':
                continue # Skip rationale nodes as primary entities
                
            if node.get('label') == basename:
                file_nodes[basename] = node
            elif file_type == 'code':
                if basename not in function_nodes:
                    function_nodes[basename] = []
                function_nodes[basename].append(node)
                
    # Document nodes
    for node in nodes:
        if node.get('file_type') == 'document':
            label = node.get('label')
            if label:
                basename = label.replace(' ', '_') + ".md"
                id_to_file[node.get('id')] = basename
                file_nodes[basename] = node

    # Build file-to-file connections
    file_links = {} # file -> set of target files
    
    for link in links:
        src = link.get('source')
        tgt = link.get('target')
        
        # We don't link rationale nodes as dependencies
        if node_by_id.get(src, {}).get('file_type') == 'rationale' or \
           node_by_id.get(tgt, {}).get('file_type') == 'rationale':
            continue
            
        src_file = id_to_file.get(src)
        tgt_file = id_to_file.get(tgt)
        
        if src_file and tgt_file and src_file != tgt_file:
            if src_file not in file_links:
                file_links[src_file] = set()
            file_links[src_file].add(tgt_file)
            
    # Ensure all files have an entry
    for file in id_to_file.values():
        if file not in file_links:
            file_links[file] = set()
            
    os.makedirs(out_dir, exist_ok=True)
    
    stems = {}
    for file in file_links.keys():
        stem = os.path.splitext(file)[0]
        if stem not in stems:
            stems[stem] = []
        stems[stem].append(file)
        
    file_to_stem = {}
    for file in file_links.keys():
        stem = os.path.splitext(file)[0]
        if len(stems[stem]) > 1:
            file_to_stem[file] = file.replace(".", "_")
        else:
            file_to_stem[file] = stem
        
    for file, targets in file_links.items():
        stem = os.path.splitext(file)[0]
        
        if len(stems[stem]) > 1:
            md_filename = file.replace(".", "_") + ".md"
            alias = file
        else:
            md_filename = stem + ".md"
            alias = file
            
        md_path = os.path.join(out_dir, md_filename)
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(f"---\naliases: [{alias}]\n---\n\n")
            f.write(f"# {file}\n\n")
            
            # File-level rationale
            file_node = file_nodes.get(file)
            if file_node:
                file_rationales = rationale_links.get(file_node.get('id'), [])
                
                # Check for other nodes that map to this file (e.g. d_1_ai_model... vs resume_scanner...)
                # Sometimes there are multiple IDs for the same file in the graph!
                if not file_rationales:
                    for node_id, node in node_by_id.items():
                        if node.get('file_type') == 'code' and node.get('label') == file:
                            file_rationales.extend(rationale_links.get(node_id, []))
                
                # Remove duplicates
                file_rationales = list(dict.fromkeys(file_rationales))
                            
                if file_rationales:
                    f.write("## Description\n")
                    for r in file_rationales:
                        f.write(f"{r}\n")
                    f.write("\n")
                    
            # Functions
            funcs = function_nodes.get(file, [])
            # Deduplicate functions by label
            seen_func_labels = set()
            unique_funcs = []
            for func in funcs:
                if func.get('label') not in seen_func_labels:
                    unique_funcs.append(func)
                    seen_func_labels.add(func.get('label'))
            
            if unique_funcs:
                f.write("## Functions\n")
                for func in unique_funcs:
                    func_label = func.get('label')
                    f.write(f"### `{func_label}`\n")
                    
                    # Gather all rationales for nodes with this function label
                    func_rationales = []
                    for f_node in funcs:
                        if f_node.get('label') == func_label:
                            func_rationales.extend(rationale_links.get(f_node.get('id'), []))
                    
                    # Remove duplicates
                    func_rationales = list(dict.fromkeys(func_rationales))
                            
                    for r in func_rationales:
                        f.write(f"> {r}\n")
                    f.write("\n")
                    
            if targets:
                f.write("## Dependencies\n")
                for tgt_file in sorted(targets):
                    tgt_stem = file_to_stem.get(tgt_file, tgt_file)
                    f.write(f"- [[{tgt_stem}]]\n")

    print(f"Generated {len(file_links)} markdown files with rich context in {out_dir}")

if __name__ == '__main__':
    main()
