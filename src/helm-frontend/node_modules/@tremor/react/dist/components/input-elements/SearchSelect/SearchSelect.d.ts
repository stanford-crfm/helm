import React from "react";
export interface SearchSelectProps extends React.HTMLAttributes<HTMLDivElement> {
    defaultValue?: string;
    value?: string;
    onValueChange?: (value: string) => void;
    placeholder?: string;
    disabled?: boolean;
    icon?: React.ElementType | React.JSXElementConstructor<any>;
    children: React.ReactElement[] | React.ReactElement;
}
declare const SearchSelect: React.ForwardRefExoticComponent<SearchSelectProps & React.RefAttributes<HTMLDivElement>>;
export default SearchSelect;
