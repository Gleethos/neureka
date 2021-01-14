package neureka.devices.storage;

import lombok.Getter;
import lombok.experimental.Accessors;
import neureka.Tsr;
import neureka.devices.Storage;
import neureka.dtype.DataType;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.lang.ref.WeakReference;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

@Accessors( prefix = {"_"} )
public class CSVHead extends AbstractFileHead<CSVHead, String>
{
    static {
        _LOG = LoggerFactory.getLogger( CSVHead.class );
    }

    @Getter private final String _delimiter;
    @Getter private final boolean _firstRowIsLabels;
    @Getter private String[] _colLabels;
    @Getter private final boolean _firstColIsIndex;
    @Getter private String[] _rowLabels;

    @Getter private Integer _numberOfRows = null;
    @Getter private Integer _numberOfColumns = null;
    private Integer _numberOfBytes = null;
    private WeakReference<String[]> _rawData = null;


    public CSVHead(
            String fileName,
            Map<String, Object> settings
    ) {
        super( fileName );
        _delimiter = (String) settings.getOrDefault( "delimiter", "," );
        _firstRowIsLabels = (boolean) settings.getOrDefault( "firstRowIsLabels", false );
        _firstColIsIndex = (boolean) settings.getOrDefault( "firstColIsIndex", false );
    }

    private String[] _lazyLoad() {
        if ( _rawData != null ) {
            String[] alreadyLoaded = _rawData.get();
            if ( alreadyLoaded != null ) return  alreadyLoaded;
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
        if ( _firstRowIsLabels ) _colLabels = table.remove( 0 );
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
        Tsr<String> loaded = new Tsr<>(
                getShape(),
                DataType.of( String.class ),
                data
        );
        String[] index;
        String[] labels;

        if ( !_firstColIsIndex ) {
            index = new String[ _numberOfRows ];
            StringBuilder prefix = new StringBuilder();
            for ( int i=0; i<index.length; i++ ) {
                int position = i % 26;
                if ( position == 25 ) prefix.append( 'a' + (i / 26) % 26 );
                index[ i ] = String.join( "", prefix.toString() + ( 'a' + position ) );
            }
        }
        else index = _rowLabels;

        if ( !_firstRowIsLabels ) {
            labels = new String[ _numberOfColumns ];
            for ( int i = 0; i < labels.length; i++ ) labels[ i ] = String.valueOf( i );
        }
        else labels = _colLabels;

        loaded.label( new String[][]{
                index,
                labels
        } );
        return loaded;
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
}
