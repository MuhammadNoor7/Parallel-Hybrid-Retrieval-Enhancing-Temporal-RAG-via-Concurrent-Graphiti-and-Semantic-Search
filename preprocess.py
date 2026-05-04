import json

def extract_shortest_temporal_reasoning(input_file, output_file, num_items=50):
    try:
        # 1. Load the original JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 2. Filter the data for 'temporal-reasoning' question types
        temporal_data = [item for item in data if item.get('question_type') == 'single-session-preference']
        
        if not temporal_data:
            print("No items with question_type='temporal-reasoning' were found.")
            return

        # 3. Define a helper function to count the number of JSON lines for an object
        def count_json_lines(item):
            # Dump the item to a formatted string to simulate how many lines it takes
            formatted_json = json.dumps(item, indent=2)
            return len(formatted_json.splitlines())
            
        # 4. Sort the filtered data based on the line count (ascending order)
        sorted_temporal_data = sorted(temporal_data, key=count_json_lines)
        
        # 5. Extract the top 50 (or fewer, if there are less than 50 total)
        shortest_50 = sorted_temporal_data[:num_items]
        
        # 6. Write the results to a new JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(shortest_50, f, indent=2, ensure_ascii=False)
            
        print(f"Successfully extracted {len(shortest_50)} shortest 'temporal-reasoning' items.")
        print(f"Results saved to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file '{input_file}' is not a valid JSON file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Define your input and output file paths
    INPUT_FILEPATH = "data/longmemeval_s_cleaned.json"
    OUTPUT_FILEPATH = "data/single_reasoning.json"
    
    # Run the extraction
    extract_shortest_temporal_reasoning(INPUT_FILEPATH, OUTPUT_FILEPATH, num_items=15)