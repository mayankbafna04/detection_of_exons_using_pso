# data/downloader.py
import os
import logging
import time
from typing import List, Optional, Tuple
from Bio import Entrez, SeqIO
from pathlib import Path

class GenBankDownloader:
    def __init__(self, email: str, api_key: Optional[str] = None, 
                 output_dir: str = "data/raw", batch_size: int = 10):
        """
        Initialize the GenBank downloader.
        
        Args:
            email: Email to identify yourself to NCBI
            api_key: NCBI API key for higher request rate limits (optional)
            output_dir: Directory to save downloaded files
            batch_size: Number of records to download in each batch
        """
        self.email = email
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
        
        # Set up Entrez
        Entrez.email = self.email
        if api_key:
            Entrez.api_key = self.api_key
            
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
    def download_genbank_files(self, accessions: List[str]) -> List[str]:
        """
        Download GenBank files for the given accession numbers.
        
        Args:
            accessions: List of GenBank accession numbers
            
        Returns:
            List of paths to downloaded files
        """
        downloaded_files = []
        
        # Process in batches to avoid overloading the NCBI server
        for i in range(0, len(accessions), self.batch_size):
            batch = accessions[i:i+self.batch_size]
            self.logger.info(f"Downloading batch {i//self.batch_size + 1}/{(len(accessions)-1)//self.batch_size + 1} ({len(batch)} accessions)")
            
            try:
                # Fetch records
                self.logger.info(f"Fetching records: {', '.join(batch)}")
                handle = Entrez.efetch(db="nucleotide", id=batch, rettype="gb", retmode="text")
                records = list(SeqIO.parse(handle, "genbank"))
                handle.close()
                
                # Save each record to a file
                for record in records:
                    accession = record.id
                    output_path = self.output_dir / f"{accession}.gb"
                    
                    with open(output_path, "w") as f:
                        SeqIO.write(record, f, "genbank")
                    
                    downloaded_files.append(str(output_path))
                    self.logger.info(f"Saved {accession} to {output_path}")
                
                # Be nice to NCBI servers
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error downloading batch {batch}: {str(e)}")
        
        return downloaded_files
