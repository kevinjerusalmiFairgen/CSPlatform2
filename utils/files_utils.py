import os
import pandas as pd
import pyreadstat


UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def save_uploaded_file(uploaded_file):
    """Saves uploaded file to the uploads folder."""
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


def load_file(file_path):
    """Loads CSV, XLSX, or SAV files into a Pandas DataFrame."""
    file_name = file_path.lower()
    metadata = {}

    try:
        if file_name.endswith(".csv"):
            df = pd.read_csv(file_path)

        elif file_name.endswith(".xlsx"):
            df = pd.read_excel(file_path, sheet_name=None)

            if len(df) == 1:
                df = list(df.values())[0]

        elif file_name.endswith(".sav"):
            df, meta = pyreadstat.read_sav(file_path)
            print(meta)

        else:
            return None, {"error": "Unsupported file type"}

        return df, meta

    except Exception as e:
        return


def save_file(df, file_path, metadata=None):
    """
    Saves a DataFrame to CSV, XLSX, or SAV format.
    """    
    file_type = file_path[-4:]
    try:
        if file_type == ".csv":
            df.to_csv(file_path, index=False)

        elif file_type == ".xlsx":
            with pd.ExcelWriter(file_path, engine="xlsxwriter") as writer:
                df.to_excel(writer, sheet_name="Sheet1", index=False)

        elif file_type == ".sav":
            pyreadstat.write_sav(df, file_path, metadata=metadata)
        else:
            print("error")
            return {"error": "Unsupported file type"}
        
        print(f"✅ File saved successfully at {file_path}")

    except Exception as e:
        return {"error": str(e)}


def empty_folder(folder_path):
    """
    Empties the contents of a folder using only the os module.
    """
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.remove(item_path)  # Remove file or symbolic link
                elif os.path.isdir(item_path):
                    os.rmdir(item_path)  # Remove empty directory
            except Exception as e:
                print(f"Error removing {item_path}: {e}")

        print(f"Folder '{folder_path}' emptied successfully.")
    else:
        print("Invalid folder path or folder does not exist.")


def get_label(metadata, column, value):
    return metadata.variable_value_labels.get(column, {}).get(value, None) 
