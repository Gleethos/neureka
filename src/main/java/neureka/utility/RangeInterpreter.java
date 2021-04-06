package neureka.utility;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import lombok.Getter;
import neureka.Tsr;
import neureka.framing.IndexAlias;
import org.slf4j.Logger;

public class RangeInterpreter {

    /**
     *  An interface provided by sl4j which enables a modular logging backend!
     */
    protected static Logger _LOG; // Why is this not final ? : For unit testing!


    private final List<Integer> _steps = new ArrayList<>();
    private final List<Object> _ranges = new ArrayList<>();

    @Getter private final int[] offset;
    //@Getter private final int[] spread;
    @Getter private final int[] newShape;


    public Object[] getRanges() {
        return _ranges.toArray();
    }

    public int[] getSteps() {
        return _steps.stream().mapToInt( s -> s ).toArray();
    }



    public RangeInterpreter( Tsr<?> source, Object[] ranges )
    {
        for (Object range : ranges ) {
            if ( range instanceof Map) {
                _ranges.addAll(((Map<?, ?>) range).keySet());
                _steps.addAll(((Map<?, Integer>) range).values());
            }
            else if ( range instanceof int[] ) {
                List<Integer> intList = new ArrayList<>(((int[]) range).length);
                for ( int ii : (int[]) range ) intList.add(ii);
                _ranges.add(intList);
                _steps.add(1);
            } else if ( range instanceof String[] ) {
                List<String> strList = new ArrayList<>(((String[]) range).length);
                strList.addAll(Arrays.asList((String[]) range));
                _ranges.add(strList);
                _steps.add(1);
            }
            else {
                _ranges.add( range );
                _steps.add(1);
            }
        }
        offset   = new int[_ranges.size()];
        newShape = new int[_ranges.size()];

        ranges = getRanges();

        for ( int i = 0; i < ranges.length; i++ ) {
            int first = 0;
            int last = 0;
            if ( !( ranges[ i ] instanceof  List ) ) {
                if ( ranges[ i ] instanceof Integer ) {
                    first = (Integer) ranges[ i ];
                    last = (Integer) ranges[ i ];
                } else {
                    IndexAlias<?> indexAlias = source.find( IndexAlias.class );
                    if ( indexAlias != null ) {
                        int position = indexAlias.get( ranges[ i ], i );
                        first = position;
                        last = position;
                    } else {
                        String message = "Given "+IndexAlias.class.getSimpleName()+" key at axis " + ( i ) + " not found!";
                        _LOG.error( message );
                        throw new IllegalStateException( message );
                    }
                }
            } else {
                ranges[ i ] = ( (List<?>) ranges[ i ] ).toArray();
                ranges[ i ] = ( ( (Object[]) ranges[ i ] )[ 0 ] instanceof List )
                        ? ( (List<?>) ( (Object[]) ranges[ i ] )[ 0 ] ).toArray()
                        : ( (Object[]) ranges[ i ] );
                if (
                        !( ( (Object[]) ( ranges[ i ] ) )[ 0 ] instanceof Integer )
                                || !( ( (Object[]) ranges[ i ] )[ ( (Object[]) ( ranges[ i ] ) ).length - 1 ] instanceof Integer )
                ) {
                    IndexAlias<?> indexAlias = source.find( IndexAlias.class );
                    if ( !( ( (Object[]) (ranges[ i ]) )[ 0 ] instanceof Integer ) ) {
                        if ( indexAlias != null ) {
                            first = indexAlias.get( ( (Object[]) ranges[ i ])[ 0 ], i );
                        }
                    }
                    else first = (Integer) ( (Object[]) ranges[ i ] )[ 0 ];

                    if ( !( ( (Object[]) ranges[ i ] )[ ( (Object[]) ranges[ i ] ).length - 1 ] instanceof Integer )  ) {
                        if ( indexAlias != null ) {
                            last = indexAlias.get(
                                    ( (Object[]) ranges[ i ] )[ ( (Object[]) ranges[ i ] ).length - 1 ],
                                    i
                            );
                        }
                    }
                    else last = (Integer) ( (Object[]) ranges[ i ] )[ ( (Object[]) ranges[ i ] ).length - 1 ];

                } else {
                    first = (Integer)( (Object[]) ranges[ i ] )[ 0 ];
                    last = (Integer) ( (Object[]) ranges[ i ] )[ ( (Object[]) ranges[ i ] ).length - 1 ];
                }
            }
            if ( first < 0 && last < 0 && first > last ) {
                int temp = first;
                first = last;
                last = temp;
            }
            first = ( first < 0 ) ? source.getNDConf().shape( i ) + first : first;
            last = ( last < 0 ) ? source.getNDConf().shape( i ) + last : last;
            newShape[ i ] = ( last - first ) + 1;
            offset[ i ] = first;
            newShape[ i ] /= _steps.get( i );
        }
    }

}
