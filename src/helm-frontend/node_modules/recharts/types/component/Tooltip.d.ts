import React, { PureComponent, CSSProperties, ReactNode, ReactElement, SVGProps } from 'react';
import { ValueType, NameType, Payload, Props as DefaultProps } from './DefaultTooltipContent';
import { AnimationDuration, AnimationTiming } from '../util/types';
export declare type ContentType<TValue extends ValueType, TName extends NameType> = ReactElement | ((props: TooltipProps<TValue, TName>) => ReactNode);
declare type UniqueFunc<TValue extends ValueType, TName extends NameType> = (entry: Payload<TValue, TName>) => unknown;
declare type UniqueOption<TValue extends ValueType, TName extends NameType> = boolean | UniqueFunc<TValue, TName>;
export declare type TooltipProps<TValue extends ValueType, TName extends NameType> = DefaultProps<TValue, TName> & {
    allowEscapeViewBox?: {
        x?: boolean;
        y?: boolean;
    };
    reverseDirection?: {
        x?: boolean;
        y?: boolean;
    };
    content?: ContentType<TValue, TName>;
    viewBox?: {
        x?: number;
        y?: number;
        width?: number;
        height?: number;
    };
    active?: boolean;
    offset?: number;
    wrapperStyle?: CSSProperties;
    cursor?: boolean | ReactElement | SVGProps<SVGElement>;
    coordinate?: {
        x?: number;
        y?: number;
    };
    position?: {
        x?: number;
        y?: number;
    };
    trigger?: 'hover' | 'click';
    shared?: boolean;
    payloadUniqBy?: UniqueOption<TValue, TName>;
    isAnimationActive?: boolean;
    animationDuration?: AnimationDuration;
    animationEasing?: AnimationTiming;
    filterNull?: boolean;
    useTranslate3d?: boolean;
};
export declare class Tooltip<TValue extends ValueType, TName extends NameType> extends PureComponent<TooltipProps<TValue, TName>> {
    static displayName: string;
    static defaultProps: {
        active: boolean;
        allowEscapeViewBox: {
            x: boolean;
            y: boolean;
        };
        reverseDirection: {
            x: boolean;
            y: boolean;
        };
        offset: number;
        viewBox: {
            x: number;
            y: number;
            height: number;
            width: number;
        };
        coordinate: {
            x: number;
            y: number;
        };
        cursorStyle: {};
        separator: string;
        wrapperStyle: {};
        contentStyle: {};
        itemStyle: {};
        labelStyle: {};
        cursor: boolean;
        trigger: string;
        isAnimationActive: boolean;
        animationEasing: string;
        animationDuration: number;
        filterNull: boolean;
        useTranslate3d: boolean;
    };
    state: {
        boxWidth: number;
        boxHeight: number;
        dismissed: boolean;
        dismissedAtCoordinate: {
            x: number;
            y: number;
        };
    };
    private wrapperNode;
    componentDidMount(): void;
    componentWillUnmount(): void;
    componentDidUpdate(): void;
    handleKeyDown: (event: KeyboardEvent) => void;
    updateBBox(): void;
    getTranslate: ({ key, tooltipDimension, viewBoxDimension, }: {
        key: 'x' | 'y';
        tooltipDimension: number;
        viewBoxDimension: number;
    }) => number;
    render(): React.JSX.Element;
}
export {};
