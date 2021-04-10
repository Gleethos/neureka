package neureka.utility.slicing;

import neureka.Tsr;
import neureka.framing.IndexAlias;
import org.slf4j.Logger;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class SmartSlicer<ValType> extends SliceBuilder<ValType>{

    /**
     *  An interface provided by sl4j which enables a modular logging backend!
     */
    protected static Logger _LOG; // Why is this not final ? : For unit testing!

    public SmartSlicer(Tsr<ValType> source, CreationCallback<ValType> callback, Object[] ranges )
    {
        super(source, callback);
        List<Object> _ranges = new ArrayList<>();
        List<Integer> _steps = new ArrayList<>();
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
            }
            else if ( range instanceof String[] ) {
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

        ranges = _ranges.toArray();

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

            this
                .axis( i )
                .from( first )
                .to( last )
                .steps( _steps.get( i ) )
                .then();

        }
    }

}
