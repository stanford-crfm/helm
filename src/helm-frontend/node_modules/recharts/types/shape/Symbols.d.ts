import React, { SVGProps } from 'react';
import { SymbolType as D3SymbolType } from 'victory-vendor/d3-shape';
import { SymbolType } from '../util/types';
declare type SizeType = 'area' | 'diameter';
interface SymbolsProp {
    className?: string;
    type: SymbolType;
    cx?: number;
    cy?: number;
    size?: number;
    sizeType?: SizeType;
}
export declare type Props = SVGProps<SVGPathElement> & SymbolsProp;
export declare const Symbols: {
    ({ type, size, sizeType, ...rest }: Props): React.JSX.Element;
    registerSymbol: (key: string, factory: D3SymbolType) => void;
};
export {};
