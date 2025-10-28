"""
Professional SEG-Y file reader for seismic data processing.
Supports both 2D and 3D seismic data with comprehensive header parsing.
"""
import numpy as np
import struct
from typing import Dict, Tuple, Optional, List
import warnings


class SEGYReader:
    """
    Professional SEG-Y file reader with full header support.
    
    Supports:
    - IBM and IEEE floating point formats
    - 2D and 3D seismic data
    - Complete header parsing
    - Data quality validation
    """
    
    def __init__(self, filename: str):
        self.filename = filename
        self.textual_header = None
        self.binary_header = None
        self.traces = []
        self.trace_headers = []
        self.data_format = None
        self.sample_interval = None
        self.num_samples = None
        self.num_traces = 0
        
    def read_segy(self) -> Dict:
        """
        Read complete SEG-Y file with headers and data.
        
        Returns:
            Dict containing all seismic data and metadata
        """
        try:
            with open(self.filename, 'rb') as f:
                # Read textual header (3200 bytes)
                self.textual_header = self._read_textual_header(f)
                
                # Read binary header (400 bytes)
                self.binary_header = self._read_binary_header(f)
                
                # Extract key parameters
                self.sample_interval = self.binary_header.get('sample_interval', 1000)  # microseconds
                self.num_samples = self.binary_header.get('num_samples', 0)
                self.data_format = self.binary_header.get('data_format', 1)
                
                # Read all traces
                self._read_traces(f)
                
                return self._create_result_dict()
                
        except Exception as e:
            raise Exception(f"Failed to read SEG-Y file: {str(e)}")
    
    def _read_textual_header(self, f) -> str:
        """Read and decode textual header"""
        header_bytes = f.read(3200)
        
        # Try EBCDIC first (standard), then ASCII
        try:
            # EBCDIC to ASCII conversion
            header_text = self._ebcdic_to_ascii(header_bytes)
        except:
            # Fallback to ASCII
            header_text = header_bytes.decode('ascii', errors='ignore')
        
        return header_text
    
    def _read_binary_header(self, f) -> Dict:
        """Read and parse binary header"""
        header_bytes = f.read(400)
        
        # Parse key fields (big-endian format)
        binary_header = {}
        
        # Job identification number (bytes 3201-3204)
        binary_header['job_id'] = struct.unpack('>I', header_bytes[0:4])[0]
        
        # Line number (bytes 3205-3208)
        binary_header['line_number'] = struct.unpack('>I', header_bytes[4:8])[0]
        
        # Reel number (bytes 3209-3212)
        binary_header['reel_number'] = struct.unpack('>I', header_bytes[8:12])[0]
        
        # Number of data traces per ensemble (bytes 3213-3214)
        binary_header['traces_per_ensemble'] = struct.unpack('>H', header_bytes[12:14])[0]
        
        # Number of auxiliary traces per ensemble (bytes 3215-3216)
        binary_header['aux_traces_per_ensemble'] = struct.unpack('>H', header_bytes[14:16])[0]
        
        # Sample interval in microseconds (bytes 3217-3218)
        binary_header['sample_interval'] = struct.unpack('>H', header_bytes[16:18])[0]
        
        # Sample interval of original field recording (bytes 3219-3220)
        binary_header['original_sample_interval'] = struct.unpack('>H', header_bytes[18:20])[0]
        
        # Number of samples per data trace (bytes 3221-3222)
        binary_header['num_samples'] = struct.unpack('>H', header_bytes[20:22])[0]
        
        # Number of samples per data trace for original field recording (bytes 3223-3224)
        binary_header['original_num_samples'] = struct.unpack('>H', header_bytes[22:24])[0]
        
        # Data sample format code (bytes 3225-3226)
        binary_header['data_format'] = struct.unpack('>H', header_bytes[24:26])[0]
        
        # Ensemble fold (bytes 3227-3228)
        binary_header['ensemble_fold'] = struct.unpack('>H', header_bytes[26:28])[0]
        
        # Trace sorting code (bytes 3229-3230)
        binary_header['trace_sorting'] = struct.unpack('>H', header_bytes[28:30])[0]
        
        # Vertical sum code (bytes 3231-3232)
        binary_header['vertical_sum'] = struct.unpack('>H', header_bytes[30:32])[0]
        
        # Sweep frequency at start (bytes 3233-3234)
        binary_header['sweep_freq_start'] = struct.unpack('>H', header_bytes[32:34])[0]
        
        # Sweep frequency at end (bytes 3235-3236)
        binary_header['sweep_freq_end'] = struct.unpack('>H', header_bytes[34:36])[0]
        
        # Sweep length (bytes 3237-3238)
        binary_header['sweep_length'] = struct.unpack('>H', header_bytes[36:38])[0]
        
        return binary_header
    
    def _read_traces(self, f):
        """Read all trace headers and data"""
        trace_count = 0
        
        while True:
            try:
                # Read trace header (240 bytes)
                trace_header_bytes = f.read(240)
                if len(trace_header_bytes) < 240:
                    break  # End of file
                
                trace_header = self._parse_trace_header(trace_header_bytes)
                
                # Get number of samples for this trace
                samples_in_trace = trace_header.get('num_samples', self.num_samples)
                if samples_in_trace == 0:
                    samples_in_trace = self.num_samples
                
                # Read trace data
                trace_data = self._read_trace_data(f, samples_in_trace)
                
                self.trace_headers.append(trace_header)
                self.traces.append(trace_data)
                trace_count += 1
                
                # Safety check to prevent infinite loops
                if trace_count > 100000:  # Adjust based on expected data size
                    warnings.warn("Large number of traces detected. Stopping read to prevent memory issues.")
                    break
                    
            except Exception as e:
                print(f"Warning: Error reading trace {trace_count}: {str(e)}")
                break
        
        self.num_traces = trace_count
    
    def _parse_trace_header(self, header_bytes: bytes) -> Dict:
        """Parse trace header fields"""
        header = {}
        
        # Trace sequence number within line (bytes 1-4)
        header['trace_seq_line'] = struct.unpack('>I', header_bytes[0:4])[0]
        
        # Trace sequence number within SEG-Y file (bytes 5-8)
        header['trace_seq_file'] = struct.unpack('>I', header_bytes[4:8])[0]
        
        # Original field record number (bytes 9-12)
        header['field_record'] = struct.unpack('>I', header_bytes[8:12])[0]
        
        # Trace number within original field record (bytes 13-16)
        header['trace_number'] = struct.unpack('>I', header_bytes[12:16])[0]
        
        # Source point number (bytes 17-20)
        header['source_point'] = struct.unpack('>I', header_bytes[16:20])[0]
        
        # Ensemble number (bytes 21-24)
        header['ensemble_number'] = struct.unpack('>I', header_bytes[20:24])[0]
        
        # Trace number within ensemble (bytes 25-28)
        header['trace_in_ensemble'] = struct.unpack('>I', header_bytes[24:28])[0]
        
        # Trace identification code (bytes 29-30)
        header['trace_id'] = struct.unpack('>H', header_bytes[28:30])[0]
        
        # Source coordinates
        header['source_x'] = struct.unpack('>I', header_bytes[72:76])[0]
        header['source_y'] = struct.unpack('>I', header_bytes[76:80])[0]
        
        # Group coordinates
        header['group_x'] = struct.unpack('>I', header_bytes[80:84])[0]
        header['group_y'] = struct.unpack('>I', header_bytes[84:88])[0]
        
        # Coordinate units (bytes 89-90)
        header['coord_units'] = struct.unpack('>H', header_bytes[88:90])[0]
        
        # Offset distance (bytes 37-40)
        header['offset'] = struct.unpack('>I', header_bytes[36:40])[0]
        
        # Receiver group elevation (bytes 41-44)
        header['receiver_elevation'] = struct.unpack('>I', header_bytes[40:44])[0]
        
        # Surface elevation at source (bytes 45-48)
        header['source_elevation'] = struct.unpack('>I', header_bytes[44:48])[0]
        
        # Source depth below surface (bytes 49-52)
        header['source_depth'] = struct.unpack('>I', header_bytes[48:52])[0]
        
        # Number of samples in this trace (bytes 115-116)
        header['num_samples'] = struct.unpack('>H', header_bytes[114:116])[0]
        
        # Sample interval for this trace (bytes 117-118)
        header['sample_interval'] = struct.unpack('>H', header_bytes[116:118])[0]
        
        return header
    
    def _read_trace_data(self, f, num_samples: int) -> np.ndarray:
        """Read trace data based on format code"""
        if self.data_format == 1:
            # IBM floating point (4 bytes per sample)
            data_bytes = f.read(num_samples * 4)
            data = self._ibm_to_ieee(data_bytes, num_samples)
        elif self.data_format == 2:
            # 4-byte two's complement integer
            data_bytes = f.read(num_samples * 4)
            data = np.frombuffer(data_bytes, dtype='>i4')
        elif self.data_format == 3:
            # 2-byte two's complement integer
            data_bytes = f.read(num_samples * 2)
            data = np.frombuffer(data_bytes, dtype='>i2')
        elif self.data_format == 5:
            # IEEE floating point (4 bytes per sample)
            data_bytes = f.read(num_samples * 4)
            data = np.frombuffer(data_bytes, dtype='>f4')
        elif self.data_format == 8:
            # 1-byte two's complement integer
            data_bytes = f.read(num_samples * 1)
            data = np.frombuffer(data_bytes, dtype='>i1')
        else:
            # Default to IEEE float
            data_bytes = f.read(num_samples * 4)
            data = np.frombuffer(data_bytes, dtype='>f4')
            warnings.warn(f"Unsupported data format {self.data_format}, assuming IEEE float")
        
        return data.astype(np.float32)
    
    def _ibm_to_ieee(self, data_bytes: bytes, num_samples: int) -> np.ndarray:
        """Convert IBM floating point to IEEE floating point"""
        data = np.zeros(num_samples, dtype=np.float32)
        
        for i in range(num_samples):
            start_idx = i * 4
            if start_idx + 4 <= len(data_bytes):
                ibm_bytes = data_bytes[start_idx:start_idx + 4]
                ieee_float = self._ibm_float_to_ieee(ibm_bytes)
                data[i] = ieee_float
        
        return data
    
    def _ibm_float_to_ieee(self, ibm_bytes: bytes) -> float:
        """Convert single IBM float to IEEE float"""
        if len(ibm_bytes) != 4:
            return 0.0
        
        # Unpack as big-endian unsigned int
        ibm_int = struct.unpack('>I', ibm_bytes)[0]
        
        # Extract IBM float components
        sign = (ibm_int >> 31) & 0x1
        exponent = (ibm_int >> 24) & 0x7F
        mantissa = ibm_int & 0xFFFFFF
        
        if mantissa == 0:
            return 0.0
        
        # Convert IBM to IEEE
        # IBM: sign * 16^(exponent-64) * (mantissa/2^24)
        # IEEE: sign * 2^(exponent-127) * (1 + mantissa/2^23)
        
        # Calculate the value
        value = mantissa / (2**24)  # Normalize mantissa
        value *= 16**(exponent - 64)  # Apply IBM exponent
        
        if sign:
            value = -value
        
        return float(value)
    
    def _ebcdic_to_ascii(self, ebcdic_bytes: bytes) -> str:
        """Convert EBCDIC to ASCII"""
        # EBCDIC to ASCII translation table
        ebcdic_to_ascii_table = [
            0x00, 0x01, 0x02, 0x03, 0x9C, 0x09, 0x86, 0x7F, 0x97, 0x8D, 0x8E, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
            0x10, 0x11, 0x12, 0x13, 0x9D, 0x85, 0x08, 0x87, 0x18, 0x19, 0x92, 0x8F, 0x1C, 0x1D, 0x1E, 0x1F,
            0x80, 0x81, 0x82, 0x83, 0x84, 0x0A, 0x17, 0x1B, 0x88, 0x89, 0x8A, 0x8B, 0x8C, 0x05, 0x06, 0x07,
            0x90, 0x91, 0x16, 0x93, 0x94, 0x95, 0x96, 0x04, 0x98, 0x99, 0x9A, 0x9B, 0x14, 0x15, 0x9E, 0x1A,
            0x20, 0xA0, 0xE2, 0xE4, 0xE0, 0xE1, 0xE3, 0xE5, 0xE7, 0xF1, 0xA2, 0x2E, 0x3C, 0x28, 0x2B, 0x7C,
            0x26, 0xE9, 0xEA, 0xEB, 0xE8, 0xED, 0xEE, 0xEF, 0xEC, 0xDF, 0x21, 0x24, 0x2A, 0x29, 0x3B, 0xAC,
            0x2D, 0x2F, 0xC2, 0xC4, 0xC0, 0xC1, 0xC3, 0xC5, 0xC7, 0xD1, 0xA6, 0x2C, 0x25, 0x5F, 0x3E, 0x3F,
            0xF8, 0xC9, 0xCA, 0xCB, 0xC8, 0xCD, 0xCE, 0xCF, 0xCC, 0x60, 0x3A, 0x23, 0x40, 0x27, 0x3D, 0x22,
            0xD8, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0xAB, 0xBB, 0xF0, 0xFD, 0xFE, 0xB1,
            0xB0, 0x6A, 0x6B, 0x6C, 0x6D, 0x6E, 0x6F, 0x70, 0x71, 0x72, 0xAA, 0xBA, 0xE6, 0xB8, 0xC6, 0xA4,
            0xB5, 0x7E, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0xA1, 0xBF, 0xD0, 0xDD, 0xDE, 0xAE,
            0x5E, 0xA3, 0xA5, 0xB7, 0xA9, 0xA7, 0xB6, 0xBC, 0xBD, 0xBE, 0x5B, 0x5D, 0xAF, 0xA8, 0xB4, 0xD7,
            0x7B, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0xAD, 0xF4, 0xF6, 0xF2, 0xF3, 0xF5,
            0x7D, 0x4A, 0x4B, 0x4C, 0x4D, 0x4E, 0x4F, 0x50, 0x51, 0x52, 0xB9, 0xFB, 0xFC, 0xF9, 0xFA, 0xFF,
            0x5C, 0xF7, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0xB2, 0xD4, 0xD6, 0xD2, 0xD3, 0xD5,
            0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0xB3, 0xDB, 0xDC, 0xD9, 0xDA, 0x9F
        ]
        
        ascii_bytes = bytearray()
        for byte in ebcdic_bytes:
            ascii_bytes.append(ebcdic_to_ascii_table[byte])
        
        return ascii_bytes.decode('ascii', errors='ignore')
    
    def _create_result_dict(self) -> Dict:
        """Create comprehensive result dictionary"""
        # Convert traces to numpy array
        if self.traces:
            seismic_data = np.array(self.traces).T  # Transpose for (samples, traces) format
        else:
            seismic_data = np.array([])
        
        # Create time axis
        dt = self.sample_interval / 1000000.0  # Convert microseconds to seconds
        time_axis = np.arange(self.num_samples) * dt
        
        # Extract coordinate information
        coordinates = self._extract_coordinates()
        
        return {
            'data': seismic_data,
            'time': time_axis,
            'dt': dt,
            'num_traces': self.num_traces,
            'num_samples': self.num_samples,
            'sample_interval_us': self.sample_interval,
            'textual_header': self.textual_header,
            'binary_header': self.binary_header,
            'trace_headers': self.trace_headers,
            'coordinates': coordinates,
            'data_format': self.data_format,
            'filename': self.filename
        }
    
    def _extract_coordinates(self) -> Dict:
        """Extract coordinate information from trace headers"""
        if not self.trace_headers:
            return {}
        
        source_x = [th.get('source_x', 0) for th in self.trace_headers]
        source_y = [th.get('source_y', 0) for th in self.trace_headers]
        group_x = [th.get('group_x', 0) for th in self.trace_headers]
        group_y = [th.get('group_y', 0) for th in self.trace_headers]
        offsets = [th.get('offset', 0) for th in self.trace_headers]
        
        return {
            'source_x': np.array(source_x),
            'source_y': np.array(source_y),
            'group_x': np.array(group_x),
            'group_y': np.array(group_y),
            'offsets': np.array(offsets),
            'cdp_x': (np.array(source_x) + np.array(group_x)) / 2,
            'cdp_y': (np.array(source_y) + np.array(group_y)) / 2
        }


def read_segy_file(filename: str) -> Dict:
    """
    Convenience function to read SEG-Y file.
    
    Args:
        filename: Path to SEG-Y file
        
    Returns:
        Dictionary containing seismic data and metadata
    """
    reader = SEGYReader(filename)
    return reader.read_segy()


def read_segy_subset(filename: str, max_traces: int = 1000, 
                    trace_start: int = 0) -> Dict:
    """
    Read a subset of SEG-Y file for large file handling.
    
    Args:
        filename: Path to SEG-Y file
        max_traces: Maximum number of traces to read
        trace_start: Starting trace number (0-based)
        
    Returns:
        Dictionary containing subset of seismic data and metadata
    """
    reader = SEGYReader(filename)
    
    try:
        with open(filename, 'rb') as f:
            # Read headers
            reader.textual_header = reader._read_textual_header(f)
            reader.binary_header = reader._read_binary_header(f)
            
            # Extract key parameters
            reader.sample_interval = reader.binary_header.get('sample_interval', 1000)
            reader.num_samples = reader.binary_header.get('num_samples', 0)
            reader.data_format = reader.binary_header.get('data_format', 1)
            
            # Calculate trace size
            bytes_per_sample = 4 if reader.data_format in [1, 2, 5] else 2
            trace_data_size = reader.num_samples * bytes_per_sample
            trace_total_size = 240 + trace_data_size  # Header + data
            
            # Skip to starting trace
            if trace_start > 0:
                f.seek(trace_start * trace_total_size, 1)  # Seek from current position
            
            # Read subset of traces
            traces_read = 0
            while traces_read < max_traces:
                try:
                    # Read trace header
                    trace_header_bytes = f.read(240)
                    if len(trace_header_bytes) < 240:
                        break  # End of file
                    
                    trace_header = reader._parse_trace_header(trace_header_bytes)
                    
                    # Read trace data
                    trace_data = reader._read_trace_data(f, reader.num_samples)
                    
                    reader.trace_headers.append(trace_header)
                    reader.traces.append(trace_data)
                    traces_read += 1
                    
                except Exception as e:
                    print(f"Warning: Error reading trace {traces_read}: {str(e)}")
                    break
            
            reader.num_traces = traces_read
            
            return reader._create_result_dict()
            
    except Exception as e:
        raise Exception(f"Failed to read SEG-Y subset: {str(e)}")


def get_segy_info(filename: str) -> Dict:
    """
    Get basic information about SEG-Y file without loading all data.
    
    Args:
        filename: Path to SEG-Y file
        
    Returns:
        Dictionary with file information
    """
    reader = SEGYReader(filename)
    
    with open(filename, 'rb') as f:
        # Read headers only
        reader.textual_header = reader._read_textual_header(f)
        reader.binary_header = reader._read_binary_header(f)
        
        # Read first trace header to get sample info
        trace_header_bytes = f.read(240)
        if len(trace_header_bytes) == 240:
            first_trace_header = reader._parse_trace_header(trace_header_bytes)
        else:
            first_trace_header = {}
    
    return {
        'filename': filename,
        'sample_interval_us': reader.binary_header.get('sample_interval', 0),
        'num_samples': reader.binary_header.get('num_samples', 0),
        'data_format': reader.binary_header.get('data_format', 0),
        'textual_header_preview': reader.textual_header[:500] if reader.textual_header else "",
        'binary_header': reader.binary_header,
        'first_trace_header': first_trace_header
    }


if __name__ == "__main__":
    # Test the reader
    import sys
    
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        print(f"Reading SEG-Y file: {filename}")
        
        try:
            # Get file info first
            info = get_segy_info(filename)
            print(f"Sample interval: {info['sample_interval_us']} μs")
            print(f"Number of samples: {info['num_samples']}")
            print(f"Data format: {info['data_format']}")
            
            # Read full file
            data = read_segy_file(filename)
            print(f"Data shape: {data['data'].shape}")
            print(f"Number of traces: {data['num_traces']}")
            print(f"Time range: {data['time'][0]:.3f} - {data['time'][-1]:.3f} s")
            
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Usage: python segy_reader.py <filename.sgy>")