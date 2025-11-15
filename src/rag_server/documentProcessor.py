import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
import ast
from itertools import zip_longest
from typing import List, Dict, Tuple

class DocumentProcessor:
    
    """Processes and splits documents into manageable text chunks."""
    
    def __init__(self, csv_path):
        """Initializes the processor with csv_path of scrapped data.

        Args:
            csv_path (str): Path to the CSV file containing scraped data.
        """
        self.csv_path = csv_path

    def get_combined_text_chunks(self):
        
        """Reads and chunks documents from given files.

        Returns:
            List[str]: List of text chunks.
        """
        
        df = pd.read_csv(self.csv_path)
        df["paragraphs"] = df["paragraphs"].apply(eval)
        df["headers"] = df["headers"].apply(eval)
        all_texts = df["paragraphs"].dropna().explode().tolist() + df["headers"].dropna().explode().tolist() # Combine paragraphs and headers
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        return text_splitter.split_text("\n\n".join(all_texts)), df["url"].iloc[0] # returns chunks and source URL
    
    def _parse_list_cell(self, x):
        """this method parses a cell that contains a lists. This is better than eval as earlier used."""
        if pd.isna(x): # If the cell is NaN, return an empty list
            return []
        if isinstance(x, list): # already a list then do nothing
            return x
        if isinstance(x, str):
            try:
                v = ast.literal_eval(x)
                if isinstance(v, list):
                    return v
                return [str(v)] # if not a list, return as single-item list (fallback)
            except (ValueError, SyntaxError):
                return [x]
        return [str(x)]

    def get_combined_text_chunks_interleaved(self):
        """
        Reads CSV rows with 'headers' and 'paragraphs' columns, interleaves them per row,
        then chunks text per row to keep provenance.

        Returns:
            chunks: List[str]
            metadatas: List[Dict]  # one metadata per chunk (url, row_idx, source='interleaved')
        """
        df = pd.read_csv(self.csv_path)

        # Ensure required columns exist
        for col in ("paragraphs", "headers", "room_numbers"):
            if col not in df.columns:
                raise KeyError(f"Missing required column: {col}")

        # Parse both columns safely into Python lists
        df["paragraphs"] = df["paragraphs"].apply(self._parse_list_cell)
        df["headers"] = df["headers"].apply(self._parse_list_cell)
        df["room_numbers"] = df["room_numbers"].apply(self._parse_list_cell)

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

        chunks: List[str] = []
        metadatas: List[Dict] = []

        # Iterate row-by-row to preserve document boundaries
        for i, row in df.iterrows():
            paras: List[str] = [p for p in row["paragraphs"] if isinstance(p, str) and p.strip()]
            heads: List[str] = [h for h in row["headers"] if isinstance(h, str) and h.strip()]
            rooms: List[str] = [r for r in row["room_numbers"] if isinstance(r, str) and r.strip()] 

            # --- Interleave strategy ---
            # We alternate: header -> paragraph -> header -> paragraph ...
            # If one list is longer, the remaining items are appended at the end.
            interleaved: List[str] = []
            for h, p, r in zip_longest(heads, paras, rooms, fillvalue=None):
                if h or h is not None:
                    interleaved.append(h.strip())
                if p or p is not None:
                    interleaved.append(p.strip())
                if r or r is not None:
                    interleaved.append(r.strip())

            # If both were empty, skip
            if not interleaved:
                continue

            doc_text = "\n\n".join(interleaved)

            # Chunk *this row's* text, this is a list of chunks for this row of type List[str]
            row_chunks = splitter.split_text(doc_text)

            # Metadata (keeps provenance for retrieval & debugging)
            url = row["url"] if "url" in df.columns and pd.notna(row["url"]) else None
            for c in row_chunks:
                chunks.append(c)
                metadatas.append({
                    "url": url,
                    "row_idx": int(i),
                    "source": "interleaved_headers_paragraphs"
                })

        return chunks, metadatas
    
    def get_rooms_text_chunks(self, rooms_csv_path):
        """
        Process rooms.csv with full_name, room_number, urls, research_info columns
        and create text chunks for office location queries.

        Args:
            rooms_csv_path: Path to the rooms.csv file

        Returns:
            chunks: List[str] - chunks containing room info, name, and research
            metadatas: List[Dict] - metadata
        """
        
        try:
            
            df = pd.read_csv(rooms_csv_path)

            chunks = []
            metadatas = []

            for i, row in df.iterrows():
                if pd.isna(row["full_name"]) or pd.isna(row["room_number"]):
                    continue

                full_name = str(row["full_name"]).strip()
                room_number = str(row["room_number"]).strip()
                research_info = str(row["research_info"]).strip() if pd.notna(row["research_info"]) else ""
                url = str(row["urls"]).strip() if pd.notna(row["urls"]) else ""

                text = [
                    f"Name: {full_name}",
                    f"Office: {room_number}",
                    f"Room: {room_number}",
                    f"Location: Room {room_number}"
                    f"Research: {research_info}"
                ]

                # Create a comprehensive text chunk for this person's office info
                text = "\n".join(text)

                # the room_text is appended to chunks list directly, because it's already a manageable size and no overlap needed
                chunks.append(text)
                metadatas.append({
                    "type": "room_info",
                    "full_name": full_name,
                    "room_number": room_number,
                    "source": url
                })

            return chunks, metadatas
        
        except Exception as e:
            print(f"Error occured in get_rooms_text_chunks: {e}")
            return [], []

    def get_combined_chunks_with_rooms(self, rooms_csv_path):
        """
        This methods combines both regular document chunks and room information chunks.

        Args:
            rooms_csv_path: Optional path to rooms.csv file

        Returns:
            chunks: List[str] - all text chunks combined
            metadatas: List[Dict] - all metadata combined
        """
        doc_chunks, doc_metadatas = self.get_combined_text_chunks_interleaved() # fetching document chunks

        room_chunks, room_metadatas = self.get_rooms_text_chunks(rooms_csv_path)
        
        # combine docs + rooms chunks and metadatas
        all_chunks = doc_chunks + room_chunks
        all_metadatas = doc_metadatas + room_metadatas
        
        return all_chunks, all_metadatas