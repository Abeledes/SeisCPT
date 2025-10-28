"""
Basic SEG-Y file reader without external dependencies.
For testing and initial file inspection.
"""
import struct
from typing import Dict, List, Tuple


def read_segy_headers(filename: str) -> Dict:
    """
    Read SEG-Y file headers without external dependencies.
    
    Args:
        filename: Path to SEG-Y file
        
    Returns:
        Dictionary with header information
    """
    try:
        with open(filename, 'rb') as f:
            # Read textual header (3200 bytes)
            textual_header_bytes = f.read(3200)
            textual_header = ebcdic_to_ascii(textual_header_bytes)
            
            # Read binary header (400 bytes)
            binary_header_bytes = f.read(400)
            binary_header = parse_binary_header(binary_header_bytes)
            
            # Read first trace header to get more info
            trace_header_bytes = f.read(240)
            if len(trace_header_bytes) == 240:
                first_trace_header = parse_trace_header(trace_header_bytes)
            else:
                first_trace_header = {}
            
            # Calculate file size and estimate number of traces
            f.seek(0, 2)  # Seek to end
            file_size = f.tell()
            
            # Calculate trace size
            num_samples = binary_header.get('num_samples', 0)
            data_format = binary_header.get('data_format', 1)
            
            bytes_per_sample = get_bytes_per_sample(data_format)
            trace_size = 240 + (num_samples * bytes_per_sample)  # Header + data
            
            # Estimate number of traces
            data_section_size = file_size - 3600  # Total - headers
            estimated_traces = data_section_size // trace_size if trace_size > 0 else 0
            
            return {
                'filename': filename,
                'file_size_bytes': file_size,
                'textual_header': textual_header,
                'binary_header': binary_header,
                'first_trace_header': first_trace_header,
                'estimated_traces': estimated_traces,
                'trace_size_bytes': trace_size,
                'data_format_name': get_data_format_name(data_format)
            }
            
    except Exception as e:
        return {'error': f"Failed to read SEG-Y file: {str(e)}"}


def parse_binary_header(header_bytes: bytes) -> Dict:
    """Parse binary header fields"""
    if len(header_bytes) < 400:
        return {}
    
    binary_header = {}
    
    try:
        # Key fields (big-endian format)
        binary_header['job_id'] = struct.unpack('>I', header_bytes[0:4])[0]
        binary_header['line_number'] = struct.unpack('>I', header_bytes[4:8])[0]
        binary_header['reel_number'] = struct.unpack('>I', header_bytes[8:12])[0]
        binary_header['traces_per_ensemble'] = struct.unpack('>H', header_bytes[12:14])[0]
        binary_header['aux_traces_per_ensemble'] = struct.unpack('>H', header_bytes[14:16])[0]
        binary_header['sample_interval'] = struct.unpack('>H', header_bytes[16:18])[0]
        binary_header['original_sample_interval'] = struct.unpack('>H', header_bytes[18:20])[0]
        binary_header['num_samples'] = struct.unpack('>H', header_bytes[20:22])[0]
        binary_header['original_num_samples'] = struct.unpack('>H', header_bytes[22:24])[0]
        binary_header['data_format'] = struct.unpack('>H', header_bytes[24:26])[0]
        binary_header['ensemble_fold'] = struct.unpack('>H', header_bytes[26:28])[0]
        binary_header['trace_sorting'] = struct.unpack('>H', header_bytes[28:30])[0]
        
    except struct.error as e:
        binary_header['parse_error'] = str(e)
    
    return binary_header


def parse_trace_header(header_bytes: bytes) -> Dict:
    """Parse trace header fields"""
    if len(header_bytes) < 240:
        return {}
    
    header = {}
    
    try:
        header['trace_seq_line'] = struct.unpack('>I', header_bytes[0:4])[0]
        header['trace_seq_file'] = struct.unpack('>I', header_bytes[4:8])[0]
        header['field_record'] = struct.unpack('>I', header_bytes[8:12])[0]
        header['trace_number'] = struct.unpack('>I', header_bytes[12:16])[0]
        header['source_point'] = struct.unpack('>I', header_bytes[16:20])[0]
        header['ensemble_number'] = struct.unpack('>I', header_bytes[20:24])[0]
        header['trace_in_ensemble'] = struct.unpack('>I', header_bytes[24:28])[0]
        header['trace_id'] = struct.unpack('>H', header_bytes[28:30])[0]
        header['offset'] = struct.unpack('>I', header_bytes[36:40])[0]
        header['source_x'] = struct.unpack('>I', header_bytes[72:76])[0]
        header['source_y'] = struct.unpack('>I', header_bytes[76:80])[0]
        header['group_x'] = struct.unpack('>I', header_bytes[80:84])[0]
        header['group_y'] = struct.unpack('>I', header_bytes[84:88])[0]
        header['coord_units'] = struct.unpack('>H', header_bytes[88:90])[0]
        header['num_samples'] = struct.unpack('>H', header_bytes[114:116])[0]
        header['sample_interval'] = struct.unpack('>H', header_bytes[116:118])[0]
        
    except struct.error as e:
        header['parse_error'] = str(e)
    
    return header


def ebcdic_to_ascii(ebcdic_bytes: bytes) -> str:
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


def get_bytes_per_sample(data_format: int) -> int:
    """Get bytes per sample based on data format code"""
    format_map = {
        1: 4,  # IBM floating point
        2: 4,  # 4-byte two's complement integer
        3: 2,  # 2-byte two's complement integer
        4: 4,  # 4-byte fixed-point with gain (obsolete)
        5: 4,  # IEEE floating point
        6: 8,  # 8-byte two's complement integer (non-standard)
        7: 3,  # 3-byte two's complement integer (non-standard)
        8: 1,  # 1-byte two's complement integer
        9: 8,  # 8-byte IEEE floating point (non-standard)
        10: 4, # 4-byte unsigned integer (non-standard)
        11: 2, # 2-byte unsigned integer (non-standard)
        12: 1  # 1-byte unsigned integer (non-standard)
    }
    return format_map.get(data_format, 4)  # Default to 4 bytes


def get_data_format_name(data_format: int) -> str:
    """Get human-readable data format name"""
    format_names = {
        1: "IBM floating point (32-bit)",
        2: "4-byte two's complement integer",
        3: "2-byte two's complement integer", 
        4: "4-byte fixed-point with gain (obsolete)",
        5: "IEEE floating point (32-bit)",
        6: "8-byte two's complement integer",
        7: "3-byte two's complement integer",
        8: "1-byte two's complement integer",
        9: "8-byte IEEE floating point",
        10: "4-byte unsigned integer",
        11: "2-byte unsigned integer",
        12: "1-byte unsigned integer"
    }
    return format_names.get(data_format, f"Unknown format ({data_format})")


def print_segy_info(filename: str):
    """Print SEG-Y file information"""
    info = read_segy_headers(filename)
    
    if 'error' in info:
        print(f"Error: {info['error']}")
        return
    
    print(f"SEG-Y File: {info['filename']}")
    print(f"File size: {info['file_size_bytes']:,} bytes")
    print()
    
    # Binary header info
    bh = info['binary_header']
    print("Binary Header:")
    print(f"  Job ID: {bh.get('job_id', 'N/A')}")
    print(f"  Line number: {bh.get('line_number', 'N/A')}")
    print(f"  Sample interval: {bh.get('sample_interval', 'N/A')} μs")
    print(f"  Number of samples: {bh.get('num_samples', 'N/A')}")
    print(f"  Data format: {bh.get('data_format', 'N/A')} ({info['data_format_name']})")
    print(f"  Traces per ensemble: {bh.get('traces_per_ensemble', 'N/A')}")
    print()
    
    # Estimated traces
    print(f"Estimated number of traces: {info['estimated_traces']}")
    print(f"Trace size: {info['trace_size_bytes']} bytes")
    print()
    
    # First trace header
    th = info['first_trace_header']
    if th:
        print("First Trace Header:")
        print(f"  Trace sequence (line): {th.get('trace_seq_line', 'N/A')}")
        print(f"  Trace sequence (file): {th.get('trace_seq_file', 'N/A')}")
        print(f"  Source point: {th.get('source_point', 'N/A')}")
        print(f"  Offset: {th.get('offset', 'N/A')}")
        print(f"  Source coordinates: ({th.get('source_x', 'N/A')}, {th.get('source_y', 'N/A')})")
        print(f"  Group coordinates: ({th.get('group_x', 'N/A')}, {th.get('group_y', 'N/A')})")
    print()
    
    # Textual header preview
    print("Textual Header (first 500 characters):")
    print("-" * 50)
    print(info['textual_header'][:500])
    print("-" * 50)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        print_segy_info(filename)
    else:
        print("Usage: python basic_segy_reader.py <filename.sgy>")