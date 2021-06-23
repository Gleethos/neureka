package neureka.devices.file.heads;

import lombok.experimental.Accessors;
import neureka.Tsr;
import neureka.devices.Storage;
import neureka.dtype.DataType;
import neureka.framing.NDFrame;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.lang.ref.WeakReference;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 *  This class is one of many extensions of the {@link AbstractFileHead} which
 *  is therefore ultimately an implementation of the {@link neureka.devices.file.FileHead} interface.
 *  Like other {@link neureka.devices.file.FileHead} implementations this class represents a file
 *  of a given type, in this case it represents a CSV file.
 */
@Accessors( prefix = {"_"} )
public class CSVHead extends AbstractFileHead<CSVHead, String>
{
    static {
        _LOG = LoggerFactory.getLogger( CSVHead.class );
    }

    private String _tensorName;
    private final String _delimiter;
    private final boolean _firstRowIsLabels;
    private String[] _colLabels;
    private final boolean _firstColIsIndex;
    private String[] _rowLabels;

    private Integer _numberOfRows = null;
    private Integer _numberOfColumns = null;
    private Integer _numberOfBytes = null;
    private WeakReference<String[]> _rawData = null;

    public CSVHead( Tsr<?> tensor, String filename )
    {
        super( filename );
        assert tensor.rank() == 2;
        _delimiter = ",";
        NDFrame<?> alias = tensor.find( NDFrame.class );
        List<Object> index = (alias != null) ? alias.atAxis( 0 ).getAllAliases() : null;
        List<Object> labels = (alias != null ) ? alias.atAxis( 1 ).getAllAliases() : null;
        _tensorName = (alias != null) ? alias.getTensorName() : null;
        _firstRowIsLabels = labels != null;
        _firstColIsIndex = index != null;
        StringBuilder asCsv = new StringBuilder();

        if ( _firstRowIsLabels ) {
            if ( _firstColIsIndex ) labels.add( 0, (_tensorName == null) ? "" : _tensorName );
            asCsv.append(
                    labels.stream().map( Object::toString ).collect( Collectors.joining(_delimiter ) )
                    + "\n"
            );
        }
        int[] shape = tensor.getNDConf().shape();
        assert shape.length == 2;
        if ( _firstColIsIndex ) assert index.size() == shape[ 0 ];
        int[] indices = new int[ 2 ];
        for ( int i = 0; i < shape[ 0 ]; i++ ) {
            indices[ 0 ] = i;
            if ( _firstColIsIndex ) asCsv.append( index.get( i ).toString() + "," );
            for ( int ii = 0; ii < shape[ 1 ]; ii++ ) {
                indices[ 1 ] = ii;
                asCsv.append( tensor.getValueAt( indices ) );
                if ( ii < shape[ 1 ] - 1 ) asCsv.append( _delimiter );
            }
            asCsv.append( "\n" );
        }
        try {
            PrintWriter out = new PrintWriter( filename );
            out.print( asCsv.toString() );
            out.close();
        } catch ( Exception e ) {
            e.printStackTrace();
        }
        tensor.setIsOutsourced( true );
        tensor.setDataType( DataType.of( String.class ) );
    }

    public CSVHead(
            String fileName,
            Map<String, Object> settings
    ) {
        super( fileName );
        if ( settings != null ) {
            _delimiter = (String) settings.getOrDefault( "delimiter", "," );
            _firstRowIsLabels = (boolean) settings.getOrDefault( "firstRowIsLabels", false );
            _firstColIsIndex = (boolean) settings.getOrDefault( "firstColIsIndex", false );
        } else {
            _delimiter = ",";
            _firstRowIsLabels = false;
            _firstColIsIndex = false;
        }
    }

    private String[] _lazyLoad() {
        if ( _rawData != null ) {
            String[] alreadyLoaded = _rawData.get();
            if ( alreadyLoaded != null ) return alreadyLoaded;
        }
        FileInputStream fis;
        try {
            fis = _loadFileInputStream();
        } catch( Exception e ) {
            e.printStackTrace();
            System.err.print( "Failed reading CSV file!" );
            _LOG.error( "Failed reading CSV file!" );
            return null;
        }
        List<String[]> table = new ArrayList<>();
        List<String> rowLabels = ( _firstColIsIndex ) ? new ArrayList<>() : null;
        try (
                BufferedReader br = new BufferedReader( new InputStreamReader( fis, StandardCharsets.UTF_8 ) )
        ) {
            String line;
            while( ( line = br.readLine() ) != null ) {
                table.add( line.split( _delimiter ) );
            }
        } catch ( IOException e ) {
            e.printStackTrace();
        }
        int rowLength = -1;
        int colHeight = 0;
        int size = 0;
        int numberOfBytes = 0;
        if ( _firstRowIsLabels ) {
            _colLabels = table.remove( 0 );
            if ( _firstColIsIndex ) {
                if ( !_colLabels[0].trim().equals("") ) _tensorName = _colLabels[0].trim();
                else _parseTensorNameFromFileName();
                String[] newLabels = new String[ _colLabels.length - 1 ];
                System.arraycopy( _colLabels, 1, newLabels, 0, newLabels.length );
                _colLabels = newLabels;
            }
            else _parseTensorNameFromFileName();
        }
        else _parseTensorNameFromFileName();

        for ( int ri = 0; ri < table.size(); ri++ ) {
            String[] row = table.get( ri );
            if ( _firstColIsIndex ) {
                rowLabels.add( row[0] );
                String[] newRow = new String[ row.length - 1 ];
                System.arraycopy( row, 1, newRow, 0, newRow.length );
                row = newRow;
                table.set( ri, newRow );
            }
            if ( rowLength < 0 ) rowLength = row.length;
            if ( rowLength == row.length ) {
                size += row.length;
                for ( String element : row )
                    numberOfBytes += element.getBytes( StandardCharsets.UTF_8 ).length;
                colHeight++;
            }
        }
        if ( rowLabels != null ) _rowLabels = rowLabels.toArray( new String[rowLabels.size()] );
        _numberOfColumns = rowLength;
        _numberOfRows = colHeight;
        _numberOfBytes = numberOfBytes;
        String[] rawData = new String[ size ];
        _rawData = new WeakReference<>( rawData );

        for ( int ri = 0; ri < _numberOfRows; ri++ ) {
            for ( int ci = 0; ci < _numberOfColumns; ci++ ) {
                rawData[ ri * rowLength + ci ] = table.get( ri )[ ci ];
            }
        }

        return rawData;
    }

    private void _parseTensorNameFromFileName() {
        String[] parts = _fileName.replace("\\", "/").split("/");
        if ( parts.length > 0 ) parts = parts[ parts.length - 1 ].split("\\.");
        _tensorName = (parts.length > 0)? parts[0] : _tensorName;
    }

    @Override
    public Storage<String> store( Tsr<String> tensor ) {
        return null;
    }

    @Override
    protected Object _loadData() throws IOException {
        return _lazyLoad();
    }

    @Override
    public Tsr<String> load() throws IOException {
        String[] data = _lazyLoad();
        Tsr<String> loaded = Tsr.of(
                getShape(),
                DataType.of( String.class ),
                data
        );
        String[] index;
        String[] labels;

        if ( !_firstColIsIndex ) {
            index = new String[ _numberOfRows ];
            for ( int i = 0; i < index.length; i++ ) index[ i ] = String.valueOf( i );
        }
        else index = _rowLabels;

        if ( !_firstRowIsLabels ) {
            labels = new String[ _numberOfColumns ];
            StringBuilder prefix = new StringBuilder( );
            for ( int i=0; i < labels.length; i++ ) {
                int position = i % 26;
                if ( position == 25 ) prefix.append( (char) ( i / 26 ) % 26 );
                labels[ i ] = String.join( "", prefix.toString() + ( (char)( 'a' + position )) );
            }
        }
        else labels = _colLabels;
        loaded.label( _tensorName,
                new String[][]{
                index,
                labels
        } );
        return loaded;
    }

    public String getTensorName() {
        _lazyLoad();
        return _tensorName;
    }

    @Override
    public int getValueSize() {
        String[] rawData;
        if ( _rawData == null ) rawData = _lazyLoad();
        else rawData = _rawData.get();
        return rawData.length;
    }

    @Override
    public int getDataSize() {
        if ( _numberOfBytes != null ) return _numberOfBytes;
        else _lazyLoad();
        return _numberOfBytes;
    }

    @Override
    public int getTotalSize() {
        return getDataSize();
    }

    @Override
    public DataType<?> getDataType() {
        return DataType.of( String.class );
    }

    @Override
    public int[] getShape() {
        return new int[]{ _numberOfRows, _numberOfColumns };
    }

    @Override
    public String extension() {
        return "csv";
    }

    public String getDelimiter() {
        return this._delimiter;
    }

    public boolean isFirstRowIsLabels() {
        return this._firstRowIsLabels;
    }

    public String[] getColLabels() {
        return this._colLabels;
    }

    public boolean isFirstColIsIndex() {
        return this._firstColIsIndex;
    }

    public String[] getRowLabels() {
        return this._rowLabels;
    }

    public Integer getNumberOfRows() {
        return this._numberOfRows;
    }

    public Integer getNumberOfColumns() {
        return this._numberOfColumns;
    }
}
