import { CartesianTickItem } from '../util/types';
import { Props as CartesianAxisProps } from './CartesianAxis';
export declare function getEveryNThTick(ticks: CartesianTickItem[]): CartesianTickItem[];
export declare function getNumberIntervalTicks(ticks: CartesianTickItem[], interval: number): CartesianTickItem[];
export declare function getTicks(props: CartesianAxisProps, fontSize?: string, letterSpacing?: string): any[];
